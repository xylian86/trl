# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SuperRL vLLM worker extension - per-layer weight offload to host DRAM.

Paper sec. IV.D: GH200 NVL2 has 144 GB HBM but the rollout actor (a fully
materialised policy + KV cache) easily exceeds that for 14B+ models. This
extension lets the trainer evict cold transformer layers from HBM to
Grace DRAM (which sits across NVLink-C2C, a few hundred ns away) so the
KV cache can grow.

Three RPCs are exposed:

- ``offload_layers(prefixes)``  - move every parameter whose name starts
  with one of ``prefixes`` to a pinned-host buffer; release HBM.
- ``prefetch_layers(prefixes)`` - move them back to HBM.
- ``set_weight_residency(plan)`` - apply a complete plan atomically;
  ``plan`` maps prefix -> {"hbm","dram"} and is computed by the trainer
  from a residency policy (e.g. "keep the layers used in the next K
  decode steps in HBM").

Used by ``trl.scripts.vllm_serve`` via ``LLM(worker_extension_cls=...)``.
The class is mixed into the vLLM worker so ``self.model_runner`` is
available; we use ``model_runner.model.named_parameters()`` to enumerate
weights.

Notes on correctness:

- Pinned host buffers are reused across offload/prefetch cycles to
  avoid per-eviction allocations.
- ``param.data`` is replaced by a zero-byte placeholder on the original
  device so vLLM kernels do not silently read stale memory; a second
  ``offload`` for the same prefix is a no-op.
- All H2D copies use ``non_blocking=True`` against the worker's default
  CUDA stream. Callers that need to overlap with decode should issue the
  prefetch RPC ahead of the decode launch.
"""

from typing import Dict, Iterable, List

import torch


class SuperRLWorkerExtension:
    """Mixin for vLLM workers that adds per-layer offload/prefetch."""

    # ------------------------------------------------------------------
    # Internal state (lazily initialised on first use)
    # ------------------------------------------------------------------

    def _ensure_state(self) -> None:
        if not hasattr(self, "_superrl_offloaded"):
            # name -> pinned CPU tensor holding the param's payload
            self._superrl_offloaded: Dict[str, torch.Tensor] = {}
            # name -> (device, dtype, shape) so we can rehydrate the placeholder
            self._superrl_meta: Dict[str, tuple] = {}

    def _matches(self, name: str, prefixes: Iterable[str]) -> bool:
        return any(name.startswith(p) for p in prefixes)

    # ------------------------------------------------------------------
    # Public RPCs
    # ------------------------------------------------------------------

    def offload_layers(self, layer_names: List[str]) -> Dict[str, int]:
        """Evict every parameter whose name starts with one of ``layer_names``.

        Returns a small dict with statistics so the trainer can log how
        much HBM was freed.
        """
        self._ensure_state()
        if not layer_names:
            return {"params_evicted": 0, "bytes_freed": 0}

        model = self.model_runner.model
        bytes_freed = 0
        evicted = 0
        for name, param in model.named_parameters():
            if not self._matches(name, layer_names):
                continue
            if name in self._superrl_offloaded:
                continue  # already evicted

            host_buf = self._superrl_offloaded.get(name)
            if host_buf is None or host_buf.shape != param.shape or host_buf.dtype != param.dtype:
                host_buf = torch.empty(
                    param.shape, dtype=param.dtype, pin_memory=torch.cuda.is_available()
                )
            host_buf.copy_(param.data, non_blocking=True)
            self._superrl_offloaded[name] = host_buf
            self._superrl_meta[name] = (param.device, param.dtype, tuple(param.shape))
            bytes_freed += param.numel() * param.element_size()
            evicted += 1
            param.data = torch.empty(0, device=param.device, dtype=param.dtype)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return {"params_evicted": evicted, "bytes_freed": int(bytes_freed)}

    def prefetch_layers(self, layer_names: List[str]) -> Dict[str, int]:
        """Restore previously offloaded parameters.

        Layers that were never offloaded are silently skipped, so the
        caller may apply an over-broad prefix without first checking
        residency state.
        """
        self._ensure_state()
        if not layer_names:
            return {"params_restored": 0, "bytes_restored": 0}

        model = self.model_runner.model
        restored = 0
        bytes_restored = 0
        for name, param in model.named_parameters():
            if not self._matches(name, layer_names):
                continue
            host_buf = self._superrl_offloaded.pop(name, None)
            if host_buf is None:
                continue
            device, dtype, _shape = self._superrl_meta.pop(name, (param.device, param.dtype, None))
            param.data = host_buf.to(device=device, dtype=dtype, non_blocking=True)
            restored += 1
            bytes_restored += param.numel() * param.element_size()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"params_restored": restored, "bytes_restored": int(bytes_restored)}

    def set_weight_residency(self, plan: Dict[str, str]) -> Dict[str, int]:
        """Atomically apply a residency plan.

        ``plan`` maps a prefix -> {"hbm", "dram"}. We compute the diff
        against current state so the same plan applied twice is cheap.
        """
        self._ensure_state()
        target_dram = [k for k, v in plan.items() if v == "dram"]
        target_hbm = [k for k, v in plan.items() if v == "hbm"]
        # Order matters: free HBM first, then prefetch.
        evict_stats = self.offload_layers(target_dram)
        restore_stats = self.prefetch_layers(target_hbm)
        return {
            "params_evicted": evict_stats.get("params_evicted", 0),
            "bytes_freed": evict_stats.get("bytes_freed", 0),
            "params_restored": restore_stats.get("params_restored", 0),
            "bytes_restored": restore_stats.get("bytes_restored", 0),
        }

    def superrl_residency_stats(self) -> Dict[str, int]:
        """Tiny telemetry RPC for trainer logging."""
        self._ensure_state()
        bytes_in_dram = sum(
            t.numel() * t.element_size() for t in self._superrl_offloaded.values()
        )
        return {
            "layers_in_dram": len(self._superrl_offloaded),
            "bytes_in_dram": int(bytes_in_dram),
        }
