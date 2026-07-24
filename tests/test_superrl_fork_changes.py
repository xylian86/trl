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

from types import SimpleNamespace

import pytest
import torch

from trl import GRPOConfig, RLOOConfig
from trl.scripts.grpo import _save_and_push_to_hub
from trl.scripts.vllm_serve import ScriptArguments as VLLMServeArguments
from trl.trainer import grpo_trainer as grpo_trainer_module
from trl.trainer.grpo_trainer import (
    GRPOTrainer,
    _generation_padding_counts,
    _length_bucket_generation_batch,
    _trim_generation_batch_padding,
)
from trl.trainer.ppo_trainer import is_reference_model_required


@pytest.mark.parametrize(
    ("kl_coef", "peft_config", "expected"),
    [
        (0.0, None, False),
        (0.0, object(), False),
        (0.05, None, True),
        (0.05, object(), False),
    ],
)
def test_reference_model_requirement(kl_coef, peft_config, expected):
    assert is_reference_model_required(kl_coef, peft_config) is expected


@pytest.mark.parametrize(
    ("save_strategy", "push_to_hub", "expected_calls"),
    [
        ("no", False, []),
        ("no", True, [("save", "output"), ("push", "dataset")]),
        ("steps", False, [("save", "output")]),
        ("steps", True, [("save", "output"), ("push", "dataset")]),
    ],
)
def test_final_save_and_hub_push_are_independent(save_strategy, push_to_hub, expected_calls):
    calls = []
    trainer = SimpleNamespace(
        save_model=lambda output_dir: calls.append(("save", output_dir)),
        push_to_hub=lambda dataset_name: calls.append(("push", dataset_name)),
    )
    args = SimpleNamespace(save_strategy=save_strategy, push_to_hub=push_to_hub, output_dir="output")

    _save_and_push_to_hub(trainer, args, "dataset")

    assert calls == expected_calls


@pytest.mark.parametrize("config_cls", [GRPOConfig, RLOOConfig])
def test_colocate_vllm_cpu_offload_must_be_nonnegative(config_cls, tmp_path):
    with pytest.raises(ValueError, match="vllm_cpu_offload_gb must be greater than or equal to 0"):
        config_cls(output_dir=str(tmp_path), vllm_cpu_offload_gb=-1.0)


def test_server_vllm_cpu_offload_must_be_nonnegative():
    with pytest.raises(ValueError, match="cpu_offload_gb must be greater than or equal to 0"):
        VLLMServeArguments(model="test-model", cpu_offload_gb=-1.0)


def test_grpo_processing_class_falls_back_to_tokenizer(monkeypatch):
    tokenizer = object()

    def no_processor(_model_id):
        raise ValueError("no processor")

    monkeypatch.setattr(grpo_trainer_module.AutoProcessor, "from_pretrained", no_processor)
    monkeypatch.setattr(grpo_trainer_module.AutoTokenizer, "from_pretrained", lambda _model_id: tokenizer)

    assert grpo_trainer_module._load_processing_class("text-only-model") is tokenizer


def test_grpo_releases_superrl_optimizer_buffers_before_rollout():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.model_wrapped = SimpleNamespace(release_superrl_optimizer_buffers=lambda: 3 * 1024**3)
    trainer.accelerator = SimpleNamespace(is_main_process=False)

    assert trainer._release_superrl_optimizer_buffers_for_rollout() == 3 * 1024**3


def test_generation_length_bucketing_reduces_padding_and_trims_batches():
    prompt_mask = torch.tensor([
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ])
    completion_mask = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
    ])
    batch = {
        "prompt_ids": torch.arange(16).reshape(4, 4),
        "prompt_mask": prompt_mask,
        "completion_ids": torch.arange(16, 32).reshape(4, 4),
        "completion_mask": completion_mask,
        "old_per_token_logps": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "advantages": torch.arange(4),
        "num_items_in_batch": torch.tensor(20),
    }

    valid, padded_before = _generation_padding_counts(batch, micro_batch_size=2)
    bucketed = _length_bucket_generation_batch(batch, micro_batch_size=2, gradient_accumulation_steps=2)
    _, padded_after = _generation_padding_counts(bucketed, micro_batch_size=2)

    assert valid.item() == 20
    assert padded_before.item() == 28
    assert padded_after.item() == 24
    assert bucketed["advantages"].tolist() == [1, 3, 2, 0]
    assert bucketed["num_items_in_batch"].item() == 20

    trimmed = _trim_generation_batch_padding({
        key: value[2:] if isinstance(value, torch.Tensor) and value.ndim else value
        for key, value in bucketed.items()
    })
    assert trimmed["prompt_ids"].shape == (2, 2)
    assert trimmed["completion_ids"].shape == (2, 2)
    assert trimmed["old_per_token_logps"].shape == (2, 2)


def test_generation_length_bucketing_stays_within_optimizer_steps():
    lengths = torch.tensor([1, 4, 2, 3, 8, 5, 7, 6])
    batch = {
        "prompt_mask": torch.ones(8, 1, dtype=torch.long),
        "completion_mask": torch.arange(8).unsqueeze(0) < lengths.unsqueeze(1),
        "sample_id": torch.arange(8),
    }

    bucketed = _length_bucket_generation_batch(batch, micro_batch_size=2, gradient_accumulation_steps=2)

    assert set(bucketed["sample_id"][:4].tolist()) == {0, 1, 2, 3}
    assert set(bucketed["sample_id"][4:].tolist()) == {4, 5, 6, 7}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger fused loss requires CUDA")
def test_liger_dapo_global_normalizer_matches_reference():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, sequence_length, hidden_size, vocab_size = 2, 5, 16, 32
    hidden_ref = torch.randn(batch_size, sequence_length, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight_ref = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype, requires_grad=True)
    token_ids = torch.randint(vocab_size, (batch_size, sequence_length), device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device=device)
    advantages = torch.tensor([0.75, -0.5], device=device)
    importance_ratio = torch.tensor([[1.0, 0.9, 1.1, 1.0, 0.8], [1.2, 1.0, 0.7, 1.0, 1.0]], device=device)

    logits = hidden_ref.float() @ weight_ref.float().t()
    logps = torch.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    old_logps = logps.detach() + 0.05 * torch.randn_like(logps)
    ratio = torch.exp(logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    per_token_loss = -torch.minimum(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
    per_token_loss = per_token_loss * importance_ratio
    num_items_in_batch = mask.sum() * 3
    reference_loss = (per_token_loss * mask).sum() / num_items_in_batch
    reference_loss.backward()

    hidden_liger = hidden_ref.detach().clone().requires_grad_(True)
    weight_liger = weight_ref.detach().clone().requires_grad_(True)
    liger_loss, _ = LigerFusedLinearGRPOLoss(
        beta=0.0, compiled=False, use_ref_model=False, chunk_size=batch_size, loss_type="dapo"
    )(
        _input=hidden_liger,
        lin_weight=weight_liger,
        selected_token_ids=token_ids,
        attention_mask=mask,
        advantages=advantages,
        old_per_token_logps=old_logps,
        vllm_is_ratio=importance_ratio,
        num_items_in_batch=num_items_in_batch,
    )
    liger_loss.backward()

    torch.testing.assert_close(liger_loss.float(), reference_loss.float(), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(hidden_liger.grad.float(), hidden_ref.grad.float(), rtol=3e-2, atol=3e-3)
    torch.testing.assert_close(weight_liger.grad.float(), weight_ref.grad.float(), rtol=3e-2, atol=3e-3)
