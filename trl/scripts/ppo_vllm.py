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
PPO with a learned critic, rule-based rewards and colocated vLLM rollouts.

    accelerate launch --config_file <deepspeed zero3 config> -m trl.scripts.ppo_vllm --config <recipe.yaml>

The critic starts from the policy checkpoint with a new scalar head; the reference policy is a frozen copy (kl_coef
> 0). Reward functions are dotted import paths called as `func(completions=[...], **dataset_columns)`.
"""

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import ModelConfig, ScriptArguments, TrlParser
from trl.trainer.ppo_vllm_config import PPOVLLMConfig
from trl.trainer.ppo_vllm_trainer import PPOVLLMTrainer


@dataclass
class PPOVLLMScriptArguments(ScriptArguments):
    reward_funcs: Optional[list[str]] = field(
        default=None, metadata={"help": "Reward functions as dotted import paths, e.g. 'my_lib.rewards.func'."}
    )


def _load_reward_funcs(names):
    funcs = []
    sys.path.insert(0, os.getcwd())
    for name in names or []:
        module_path, func_name = name.rsplit(".", 1)
        funcs.append(getattr(importlib.import_module(module_path), func_name))
    if not funcs:
        raise ValueError("PPO needs at least one reward function (reward_funcs).")
    return funcs


def _zero_value_head(critic):
    """Start the critic's new value head at zero. With a random head the first values are in the hundreds (large
    hidden-state norms), and its loss would dominate the gradient norm that the shared optimizer clips."""
    params = list(critic.score.parameters())
    if any(hasattr(p, "ds_id") for p in params):  # partitioned by ZeRO-3 at load
        import deepspeed

        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                for p in params:
                    p.data.zero_()
    else:
        for p in params:
            p.data.zero_()


def main(script_args, training_args, model_args):
    path = model_args.model_name_or_path
    kwargs = dict(training_args.model_init_kwargs or {})
    for key in ("dtype", "torch_dtype"):
        if isinstance(kwargs.get(key), str) and kwargs[key] != "auto":
            kwargs[key] = getattr(torch, kwargs[key])
    if model_args.attn_implementation is not None:
        kwargs["attn_implementation"] = model_args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=model_args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=model_args.trust_remote_code, **kwargs)
    critic = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **kwargs)
    critic.config.pad_token_id = tokenizer.pad_token_id
    _zero_value_head(critic)
    ref_policy = None
    if training_args.kl_coef != 0.0:
        ref_policy = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=model_args.trust_remote_code,
                                                          **kwargs)

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    trainer = PPOVLLMTrainer(
        args=training_args,
        processing_class=tokenizer,
        policy=policy,
        critic=critic,
        ref_policy=ref_policy,
        reward_funcs=_load_reward_funcs(script_args.reward_funcs),
        train_dataset=dataset[script_args.dataset_train_split],
    )
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((PPOVLLMScriptArguments, PPOVLLMConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
