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

from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class PPOVLLMConfig(TrainingArguments):
    r"""
    Configuration for [`PPOVLLMTrainer`]: PPO with a learned critic (GAE), rule-based rewards and rollouts on a vLLM
    engine colocated with the trainer.

    Batch sizes are per device and count sequences: `per_device_train_batch_size` sequences per micro-batch,
    `gradient_accumulation_steps` micro-batches per optimizer step (one mini-batch), `num_mini_batches` mini-batches
    per rollout, so each rollout generates `per_device_train_batch_size * gradient_accumulation_steps *
    num_mini_batches` sequences per device, `num_generations` per prompt. `max_steps` counts rollouts.

    The defaults follow veRL's PPO recipe: GAE with gamma = lambda = 1, KL penalty in the reward (k1, 0.001), policy
    clip 0.2, value clip 0.5, one PPO epoch over four mini-batches, critic learning rate 1e-5. The policy and the
    critic share one optimizer with one parameter group each, so `learning_rate` applies to the policy and
    `critic_learning_rate` to the critic; `vf_coef` weights the value loss.
    """

    learning_rate: float = field(default=5e-7, metadata={"help": "Policy learning rate."})
    critic_learning_rate: float = field(default=1e-5, metadata={"help": "Critic learning rate."})
    num_generations: int = field(default=8, metadata={"help": "Completions sampled per prompt."})
    max_prompt_length: int = field(default=1024, metadata={"help": "Prompts are truncated on the left to this."})
    max_completion_length: int = field(default=512, metadata={"help": "Maximum generated tokens per completion."})
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature, also used for log-probs."})
    top_p: float = field(default=1.0, metadata={"help": "Nucleus sampling."})
    num_ppo_epochs: int = field(default=1, metadata={"help": "Passes over each rollout batch."})
    num_mini_batches: int = field(default=4, metadata={"help": "Optimizer steps per pass over a rollout batch."})
    local_rollout_forward_batch_size: int = field(
        default=32, metadata={"help": "Sequences per forward pass when scoring a rollout (no gradients)."}
    )
    kl_coef: float = field(default=0.001, metadata={"help": "KL penalty coefficient in the reward; 0 drops the ref model."})
    kl_estimator: str = field(default="k1", metadata={"help": "'k1' or 'k3'."})
    cliprange: float = field(default=0.2, metadata={"help": "PPO ratio clip."})
    cliprange_value: float = field(default=0.5, metadata={"help": "Value clip around the rollout values."})
    vf_coef: float = field(default=0.1, metadata={"help": "Value loss weight."})
    gamma: float = field(default=1.0, metadata={"help": "Discount factor."})
    lam: float = field(default=1.0, metadata={"help": "GAE lambda."})
    whiten_advantages: bool = field(default=True, metadata={"help": "Normalize advantages over the rollout batch."})
    vllm_gpu_memory_utilization: float = field(default=0.66, metadata={"help": "Fraction of HBM for vLLM."})
    vllm_tensor_parallel_size: int = field(default=1, metadata={"help": "Tensor parallel size of each vLLM engine."})
    vllm_enable_sleep_mode: bool = field(default=True, metadata={"help": "Release vLLM's memory between rollouts."})
    vllm_model_impl: str = field(default="vllm", metadata={"help": "'vllm' or 'transformers'."})
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None, metadata={"help": "Keyword arguments for `from_pretrained` of the policy, ref and critic."}
    )
    logging_steps: float = field(default=1, metadata={"help": "Log every this many rollouts."})
    bf16: Optional[bool] = field(default=None, metadata={"help": "Use bf16 (defaults to True unless fp16 is set)."})

    def __post_init__(self):
        self.bf16 = not self.fp16 if self.bf16 is None else self.bf16
        super().__post_init__()
        if self.kl_estimator not in ("k1", "k3"):
            raise ValueError(f"kl_estimator must be 'k1' or 'k3', got {self.kl_estimator!r}.")
        local_batch = self.per_device_train_batch_size * self.gradient_accumulation_steps * self.num_mini_batches
        if local_batch % self.num_generations:
            raise ValueError(f"Sequences per device per rollout ({local_batch}) must be a multiple of "
                             f"num_generations ({self.num_generations}).")
