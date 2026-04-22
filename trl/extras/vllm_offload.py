# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for SuperRL vLLM weight-offload integration in trainers.

Used by GRPO, RLOO, and Online DPO trainers. Each trainer holds:

- ``self.use_vllm`` and ``self.vllm_mode`` ('server' | 'colocate')
- ``self.vllm_client`` (server mode) or ``self.llm`` (colocate mode)
- ``self.model`` (the policy)
- ``self.args.vllm_weight_offload``,
  ``self.args.vllm_weight_offload_active_layers``,
  ``self.args.vllm_weight_offload_use_c2c``
- ``self.accelerator.is_main_process``

The helpers below assume those attributes are present.
"""

from typing import Dict, List


def vllm_call(trainer, method: str, args: list) -> None:
    """Dispatch a vLLM worker-extension RPC in either deployment mode.

    - ``server`` mode -> HTTP via ``trainer.vllm_client.<method>(*args)``
    - ``colocate`` mode -> in-process via ``trainer.llm.collective_rpc(...)``
    """
    if not getattr(trainer, "use_vllm", False):
        return
    mode = getattr(trainer, "vllm_mode", None)
    if mode == "server":
        client_method = getattr(trainer.vllm_client, method, None)
        if client_method is not None:
            client_method(*args)
    elif mode == "colocate":
        trainer.llm.collective_rpc(method=method, args=tuple(args))


def build_weight_residency_plan(trainer) -> Dict[str, str]:
    """Return a layer-name -> ``"hbm"|"dram"`` residency plan.

    Selects the last ``active_layers`` transformer blocks for HBM residency
    (deeper layers tend to be re-touched soonest in the autoregressive
    decode path), and offloads the rest to host DRAM.
    """
    model = trainer.model
    layer_names: List[str] = []
    for name, _ in model.named_parameters():
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            prefix = ".".join(parts[: idx + 2])
            if prefix not in layer_names:
                layer_names.append(prefix)

    if not layer_names:
        return {}

    active = trainer.args.vllm_weight_offload_active_layers
    if isinstance(active, float) and active <= 1.0:
        n_active = max(1, int(len(layer_names) * active))
    else:
        n_active = max(1, int(active))

    hbm_set = set(layer_names[-n_active:])
    return {ln: ("hbm" if ln in hbm_set else "dram") for ln in layer_names}


def apply_residency_after_weight_sync(trainer) -> None:
    """Call after ``_move_model_to_vllm``: install plan and stash on trainer."""
    if not getattr(trainer.args, "vllm_weight_offload", False):
        return
    if not trainer.accelerator.is_main_process:
        return
    plan = build_weight_residency_plan(trainer)
    vllm_call(trainer, "set_weight_residency", [plan])
    trainer._superrl_offload_plan = plan


def prefetch_before_rollout(trainer) -> None:
    """Call before vLLM generation: hoist HBM-tier layers back to HBM."""
    if not getattr(trainer.args, "vllm_weight_offload", False):
        return
    if not trainer.accelerator.is_main_process:
        return
    plan = getattr(trainer, "_superrl_offload_plan", {})
    hbm_layers = [k for k, v in plan.items() if v == "hbm"]
    if hbm_layers:
        vllm_call(trainer, "prefetch_layers", [hbm_layers])


def offload_after_rollout(trainer) -> None:
    """Call after vLLM generation: evict DRAM-tier layers back to host."""
    if not getattr(trainer.args, "vllm_weight_offload", False):
        return
    if not trainer.accelerator.is_main_process:
        return
    plan = getattr(trainer, "_superrl_offload_plan", {})
    dram_layers = [k for k, v in plan.items() if v == "dram"]
    if dram_layers:
        vllm_call(trainer, "offload_layers", [dram_layers])
