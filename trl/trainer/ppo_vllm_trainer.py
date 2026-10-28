# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO with a learned critic, rule-based rewards and rollouts on a colocated vLLM engine, trained with DeepSpeed ZeRO-3
through accelerate.

One update: wake vLLM, load the policy's weights into it, sample `num_generations` completions per prompt, put vLLM to
sleep; score the rollout without gradients (policy log-probs, reference log-probs, critic values, rule rewards); form
per-token rewards (KL penalty plus the score on the last token) and GAE advantages; then `num_ppo_epochs` passes of
`num_mini_batches` optimizer steps over the policy and the critic, which share one DeepSpeed engine.
"""

import faulthandler
import os
import sys
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..core import masked_mean, masked_whiten
from ..import_utils import is_vllm_available
from ..models.utils import prepare_deepspeed
from .ppo_vllm_config import PPOVLLMConfig
from .utils import disable_dropout_in_model, empty_cache


if is_vllm_available():
    from vllm import LLM, SamplingParams

logger = get_logger(__name__)


def _token_logprobs(logits: torch.Tensor, tokens: torch.Tensor, temperature: float) -> torch.Tensor:
    """Log-probabilities of `tokens` under softmax(logits / temperature), in fp32, one row at a time."""
    rows = []
    for row_logits, row_tokens in zip(logits, tokens):
        row = torch.log_softmax(row_logits.float() / temperature, dim=-1)
        rows.append(row.gather(-1, row_tokens.unsqueeze(-1)).squeeze(-1))
    return torch.stack(rows)


def _policy_logits(policy: nn.Module, input_ids, attention_mask, position_ids, response_length: int) -> torch.Tensor:
    """Logits that predict the last `response_length` tokens."""
    out = policy(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False,
                 logits_to_keep=response_length + 1)
    return out.logits[:, :-1]


def _critic_values(critic: nn.Module, input_ids, attention_mask, position_ids, response_length: int) -> torch.Tensor:
    """V(s_t) for each of the last `response_length` tokens: the value head on the hidden state before the token."""
    backbone = getattr(critic, critic.base_model_prefix)
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                      use_cache=False).last_hidden_state
    return critic.score(hidden[:, -response_length - 1:-1]).squeeze(-1).float()


class PolicyAndCritic(nn.Module):
    """The policy and the critic as one module, so that one DeepSpeed engine (one optimizer) trains both."""

    def __init__(self, policy: PreTrainedModel, critic: PreTrainedModel):
        super().__init__()
        self.policy = policy
        self.critic = critic

    def forward(self, input_ids, attention_mask, position_ids, response_length: int):
        logits = _policy_logits(self.policy, input_ids, attention_mask, position_ids, response_length)
        values = _critic_values(self.critic, input_ids, attention_mask, position_ids, response_length)
        return logits, values


class PPOVLLMTrainer:
    """PPO (actor-critic, GAE) with rule-based rewards and colocated vLLM rollouts. See [`PPOVLLMConfig`]."""

    def __init__(
        self,
        args: PPOVLLMConfig,
        processing_class: PreTrainedTokenizerBase,
        policy: PreTrainedModel,
        critic: PreTrainedModel,
        ref_policy: Optional[PreTrainedModel],
        reward_funcs: list[Callable],
        train_dataset: Dataset,
    ):
        if not is_vllm_available():
            raise ImportError("PPOVLLMTrainer needs vLLM.")
        if args.kl_coef != 0.0 and ref_policy is None:
            raise ValueError("kl_coef is nonzero, so a reference policy is required.")
        self.args = args
        self.tokenizer = processing_class
        self.reward_funcs = reward_funcs
        self.accelerator = accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        # `accelerate launch` exports ACCELERATE_GRADIENT_ACCUMULATION_STEPS (1 by default); the DeepSpeed plugin
        # fills the config's "auto" accumulation from it at creation, and prepare() would fill the batch fields from
        # the prompt dataloader. The PPO batch sizes define all three, so set them here: the engine then steps the
        # optimizer once per mini-batch.
        accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
        if accelerator.state.deepspeed_plugin is not None:
            plugin = accelerator.state.deepspeed_plugin
            plugin.gradient_accumulation_steps = args.gradient_accumulation_steps
            plugin.deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
            plugin.deepspeed_config["train_batch_size"] = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes)
        self.mini_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.rollout_batch_size = self.mini_batch_size * args.num_mini_batches
        self.prompts_per_rollout = self.rollout_batch_size // args.num_generations
        self.pad_token_id = processing_class.pad_token_id
        eos = policy.generation_config.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos]) | {processing_class.eos_token_id}

        for model in (policy, critic, ref_policy):
            if model is not None:
                disable_dropout_in_model(model)
                model.config.use_cache = False
        if args.gradient_checkpointing:
            kwargs = args.gradient_checkpointing_kwargs or {"use_reentrant": False}
            policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
            critic.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        if args.use_liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            for model in (policy, critic, ref_policy):
                if model is not None:
                    _apply_liger_kernel_to_instance(model=model)

        self._init_vllm(policy.name_or_path)

        self.model = PolicyAndCritic(policy, critic)
        optimizer = torch.optim.AdamW(
            [{"params": [p for p in policy.parameters() if p.requires_grad], "lr": args.learning_rate},
             {"params": [p for p in critic.parameters() if p.requires_grad], "lr": args.critic_learning_rate}],
            betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon, weight_decay=args.weight_decay)
        dataloader = DataLoader(train_dataset, batch_size=self.prompts_per_rollout, shuffle=True, drop_last=True,
                                collate_fn=lambda rows: {key: [row[key] for row in rows] for key in rows[0]})
        torch.manual_seed(args.seed)  # the same shuffle on every rank before accelerate shards the batches
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, optimizer, dataloader)
        torch.manual_seed(args.seed + accelerator.process_index)
        self.ref_policy = prepare_deepspeed(ref_policy, accelerator) if ref_policy is not None else None
        self.num_updates = args.max_steps if args.max_steps > 0 else \
            int(args.num_train_epochs * len(self.dataloader))
        # PPO_HANG_DUMP_S: if no rollout, scoring chunk or micro-batch starts for this many seconds, faulthandler's
        # watchdog thread prints every thread's Python stack to stderr (it needs neither the GIL nor a signal).
        self.hang_dump_s = float(os.environ.get("PPO_HANG_DUMP_S", "600"))

    def _arm_watchdog(self):
        if self.hang_dump_s > 0:
            faulthandler.dump_traceback_later(self.hang_dump_s, repeat=True, file=sys.stderr)

    # --------------------------------------------------------------------------------------------- vLLM rollouts
    def _init_vllm(self, model_path: str):
        args, accelerator = self.args, self.accelerator
        tp = args.vllm_tensor_parallel_size
        if accelerator.num_processes % tp:
            raise ValueError(f"vllm_tensor_parallel_size ({tp}) must divide the world size "
                             f"({accelerator.num_processes}).")
        self.tp_group = None
        if tp > 1:
            self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                [list(range(i * tp, (i + 1) * tp)) for i in range(accelerator.num_processes // tp)])
        os.environ["RANK"] = str(accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12345")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_num_seqs=self.rollout_batch_size * tp,
            max_model_len=args.max_prompt_length + args.max_completion_length,
            distributed_executor_backend="external_launcher",
            seed=accelerator.process_index // tp,  # the ranks of a TP group must sample alike
            max_num_batched_tokens=4096,
            model_impl=args.vllm_model_impl,
            enable_sleep_mode=args.vllm_enable_sleep_mode,
        )
        if args.vllm_enable_sleep_mode:
            self.llm.sleep(level=1)
        accelerator.wait_for_everyone()

    def _release_superrl_optimizer_buffers_for_rollout(self):
        # SuperRL-IO keeps its optimizer swap buffers in HBM; release them while vLLM holds the memory, unless
        # SUPERRL_KEEP_OPTIMIZER_BUFFERS=1 (then vLLM must leave room for them).
        if os.environ.get("SUPERRL_KEEP_OPTIMIZER_BUFFERS") == "1":
            return
        release_fn = getattr(self.model, "release_superrl_optimizer_buffers", None)
        if release_fn is not None:
            release_fn()

    def _load_policy_into_vllm(self):
        import deepspeed

        vllm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        policy = self.accelerator.unwrap_model(self.model).policy
        for name, param in policy.named_parameters():
            with deepspeed.zero.GatheredParameters([param]):
                vllm_model.load_weights([(name.replace("_checkpoint_wrapped_module.", ""), param.data)])
        self.llm.reset_prefix_cache()

    def _generate(self, prompt_ids: list[list[int]]) -> list[list[int]]:
        args = self.args
        self._arm_watchdog()
        if args.vllm_enable_sleep_mode:
            self._release_superrl_optimizer_buffers_for_rollout()
            torch.cuda.empty_cache()
            self.llm.wake_up()
        self._load_policy_into_vllm()
        tp = args.vllm_tensor_parallel_size
        requests = prompt_ids
        if tp > 1:  # every rank of a TP group generates for the whole group, then keeps its own share
            gathered = [None] * tp
            torch.distributed.all_gather_object(gathered, prompt_ids, group=self.tp_group)
            requests = [ids for part in gathered for ids in part]
        sampling = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                                  max_tokens=args.max_completion_length)
        outputs = self.llm.generate([{"prompt_token_ids": ids} for ids in requests], sampling_params=sampling,
                                    use_tqdm=False)
        completions = [list(out.outputs[0].token_ids) for out in outputs]
        if tp > 1:
            rank = torch.distributed.get_rank(group=self.tp_group)
            completions = completions[rank * len(prompt_ids):(rank + 1) * len(prompt_ids)]
        if args.vllm_enable_sleep_mode:
            self.llm.sleep(level=1)
        return completions

    # ------------------------------------------------------------------------------------------------ batching
    def _make_batch(self, prompts: list[str]) -> dict:
        """Tokenize prompts (left-padded, left-truncated), generate, and right-pad the completions."""
        args, device = self.args, self.accelerator.device
        encoded = [self.tokenizer(p, add_special_tokens=False)["input_ids"][-args.max_prompt_length:] for p in prompts]
        completions = self._generate(encoded)
        lengths = []
        for ids in completions:  # keep tokens up to and including the first EOS
            end = next((i + 1 for i, t in enumerate(ids) if t in self.eos_token_ids), len(ids))
            lengths.append(max(end, 1))
        prompt_len = max(len(ids) for ids in encoded)
        response_len = max(lengths)
        batch = len(prompts)
        query = torch.full((batch, prompt_len), self.pad_token_id, dtype=torch.long)
        query_mask = torch.zeros((batch, prompt_len), dtype=torch.long)
        response = torch.full((batch, response_len), self.pad_token_id, dtype=torch.long)
        response_mask = torch.zeros((batch, response_len), dtype=torch.long)
        for i, (ids, comp, n) in enumerate(zip(encoded, completions, lengths)):
            query[i, prompt_len - len(ids):] = torch.tensor(ids)
            query_mask[i, prompt_len - len(ids):] = 1
            comp = (comp[:n] or [self.pad_token_id])
            response[i, :len(comp)] = torch.tensor(comp)
            response_mask[i, :n] = 1
        input_ids = torch.cat([query, response], 1).to(device)
        attention_mask = torch.cat([query_mask, response_mask], 1).to(device)
        position_ids = (attention_mask.cumsum(1) - 1).clamp(min=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
                "responses": response.to(device), "response_mask": response_mask.to(device).bool(),
                "response_lengths": torch.tensor(lengths, device=device), "completions": completions}

    @torch.no_grad()
    def _score(self, batch: dict) -> dict:
        """Rollout log-probs of the policy and the reference, and critic values, without gradients."""
        args = self.args
        policy = self.accelerator.unwrap_model(self.model).policy
        critic = self.accelerator.unwrap_model(self.model).critic
        response_len = batch["responses"].shape[1]
        logprobs, ref_logprobs, values = [], [], []
        for start in range(0, batch["input_ids"].shape[0], args.local_rollout_forward_batch_size):
            self._arm_watchdog()
            sl = slice(start, start + args.local_rollout_forward_batch_size)
            inputs = (batch["input_ids"][sl], batch["attention_mask"][sl], batch["position_ids"][sl])
            logits = _policy_logits(policy, *inputs, response_len)
            logprobs.append(_token_logprobs(logits, batch["responses"][sl], args.temperature))
            del logits
            if self.ref_policy is not None:
                logits = _policy_logits(self.ref_policy, *inputs, response_len)
                ref_logprobs.append(_token_logprobs(logits, batch["responses"][sl], args.temperature))
                del logits
            values.append(_critic_values(critic, *inputs, response_len))
        out = {"logprobs": torch.cat(logprobs), "values": torch.cat(values)}
        out["ref_logprobs"] = torch.cat(ref_logprobs) if ref_logprobs else out["logprobs"]
        return out

    def _rewards(self, completions: list[list[int]], extra: dict) -> torch.Tensor:
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        total = torch.zeros(len(texts), device=self.accelerator.device)
        for func in self.reward_funcs:
            scores = func(completions=texts, **extra)
            total += torch.tensor([0.0 if s is None else float(s) for s in scores], device=total.device)
        return total

    def _advantages(self, batch: dict, scored: dict, scores: torch.Tensor):
        """Per-token rewards (KL penalty, plus the score on the last token), then GAE advantages and returns."""
        args = self.args
        mask = batch["response_mask"]
        logr = scored["ref_logprobs"] - scored["logprobs"]
        kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr
        kl = kl.masked_fill(~mask, 0.0)
        rewards = -args.kl_coef * kl
        last = batch["response_lengths"] - 1
        rewards[torch.arange(rewards.shape[0], device=rewards.device), last] += scores
        values = scored["values"].masked_fill(~mask, 0.0)
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(rewards.shape[0], device=rewards.device)
        for t in reversed(range(rewards.shape[1])):
            next_values = values[:, t + 1] if t + 1 < rewards.shape[1] else 0.0
            delta = rewards[:, t] + args.gamma * next_values - values[:, t]
            lastgaelam = delta + args.gamma * args.lam * lastgaelam
            advantages[:, t] = lastgaelam
        returns = advantages + values
        if args.whiten_advantages:
            advantages = masked_whiten(advantages, mask)
        return advantages.masked_fill(~mask, 0.0), returns, kl

    # ------------------------------------------------------------------------------------------------- training
    def train(self):
        args, accelerator = self.args, self.accelerator
        self.model.train()
        batches = iter(self._repeat(self.dataloader))
        for update in range(1, self.num_updates + 1):
            t0 = time.perf_counter()
            data = next(batches)
            prompts = [p for p in data["prompt"] for _ in range(args.num_generations)]
            extra = {k: [v for v in vals for _ in range(args.num_generations)] for k, vals in data.items()
                     if k != "prompt"}
            batch = self._make_batch(prompts)
            torch.cuda.synchronize()
            t_rollout = time.perf_counter() - t0
            scored = self._score(batch)
            scores = self._rewards(batch["completions"], extra)
            advantages, returns, kl = self._advantages(batch, scored, scores)
            torch.cuda.synchronize()
            t_score = time.perf_counter() - t0 - t_rollout
            stats = self._update(batch, scored, advantages, returns)
            torch.cuda.synchronize()
            t_total = time.perf_counter() - t0
            if update % max(1, int(args.logging_steps)) == 0:
                self._log(update, t_total, t_rollout, t_score, batch, scores, kl, stats)
        faulthandler.cancel_dump_traceback_later()

    @staticmethod
    def _repeat(dataloader):
        while True:
            yield from dataloader

    def _update(self, batch: dict, scored: dict, advantages: torch.Tensor, returns: torch.Tensor) -> dict:
        args, accelerator = self.args, self.accelerator
        mask = batch["response_mask"]
        response_len = batch["responses"].shape[1]
        stats = {k: [] for k in ("pg_loss", "vf_loss", "pg_clipfrac", "vf_clipfrac", "approx_kl", "ratio")}
        n = batch["input_ids"].shape[0]
        micro = args.per_device_train_batch_size
        for _ in range(args.num_ppo_epochs):
            order = torch.randperm(n, device=mask.device)
            for mb_start in range(0, n, self.mini_batch_size):
                mb = order[mb_start:mb_start + self.mini_batch_size]
                for mi_start in range(0, len(mb), micro):
                    self._arm_watchdog()
                    idx = mb[mi_start:mi_start + micro]
                    m = mask[idx]
                    # accumulate(): only the last micro-batch of a mini-batch is a gradient-accumulation boundary,
                    # so the engine steps the optimizer once per mini-batch.
                    with accelerator.accumulate(self.model):
                        logits, vpred = self.model(input_ids=batch["input_ids"][idx],
                                                   attention_mask=batch["attention_mask"][idx],
                                                   position_ids=batch["position_ids"][idx],
                                                   response_length=response_len)
                        new_logprobs = _token_logprobs(logits, batch["responses"][idx], args.temperature)
                        del logits
                        log_ratio = (new_logprobs - scored["logprobs"][idx]).masked_fill(~m, 0.0)
                        ratio = log_ratio.exp()
                        adv = advantages[idx]
                        pg_losses = torch.max(-adv * ratio,
                                              -adv * ratio.clamp(1 - args.cliprange, 1 + args.cliprange))
                        pg_loss = masked_mean(pg_losses, m)
                        old_values = scored["values"][idx]
                        vclipped = old_values + (vpred - old_values).clamp(-args.cliprange_value,
                                                                           args.cliprange_value)
                        vf_losses = torch.max((vpred - returns[idx]) ** 2, (vclipped - returns[idx]) ** 2)
                        vf_loss = 0.5 * masked_mean(vf_losses, m)
                        loss = pg_loss + args.vf_coef * vf_loss
                        accelerator.backward(loss)
                    with torch.no_grad():
                        stats["pg_loss"].append(pg_loss)
                        stats["vf_loss"].append(vf_loss)
                        stats["pg_clipfrac"].append(masked_mean(((ratio - 1).abs() > args.cliprange).float(), m))
                        stats["vf_clipfrac"].append(masked_mean(
                            ((vpred - old_values).abs() > args.cliprange_value).float(), m))
                        stats["approx_kl"].append(masked_mean(0.5 * log_ratio ** 2, m))
                        stats["ratio"].append(masked_mean(ratio, m))
        empty_cache()
        return {k: torch.stack(v).mean() for k, v in stats.items()}

    def _log(self, update, t_total, t_rollout, t_score, batch, scores, kl, stats):
        accelerator = self.accelerator
        mask = batch["response_mask"]
        local = torch.stack([scores.mean(), kl.sum(1).mean(), mask.sum(1).float().mean(),
                             mask.sum().float(), *stats.values()])
        mean = accelerator.reduce(local, reduction="mean")
        tokens = accelerator.reduce(batch["attention_mask"].sum().float(), reduction="sum")
        values = {"step": update, "step_time": t_total, "rollout_time": t_rollout, "score_time": t_score,
                  "update_time": t_total - t_rollout - t_score, "reward": mean[0].item(), "kl": mean[1].item(),
                  "completions/mean_length": mean[2].item(), "num_tokens": tokens.item(),
                  "optimizer_steps": getattr(self.model, "global_steps", None),
                  "gradient_accumulation_steps": getattr(self.model, "gradient_accumulation_steps", lambda: None)(),
                  **{k: mean[4 + i].item() for i, k in enumerate(stats)}}
        if accelerator.is_main_process:
            print({"ppo": values}, flush=True)
