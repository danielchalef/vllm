# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static NVFP4 MoE backend-selection policy tests."""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle import nvfp4
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)


def _make_moe_config(moe_backend: str = "auto") -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=2048,
        intermediate_size_per_partition=512,
        num_local_experts=256,
        num_logical_experts=256,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


class _AlwaysSupportedExperts:
    @staticmethod
    def is_supported_config(
        cls,
        moe_config: FusedMoEConfig,
        weight_key,
        activation_key,
        activation_format: FusedMoEActivationFormat,
    ):
        return True, None


class _W4A4OnlyExperts:
    @staticmethod
    def is_supported_config(
        cls,
        moe_config: FusedMoEConfig,
        weight_key,
        activation_key,
        activation_format: FusedMoEActivationFormat,
    ):
        if activation_key is None:
            return False, "requires activation quantization"
        return True, None


class _W4A16Experts:
    @staticmethod
    def is_supported_config(
        cls,
        moe_config: FusedMoEConfig,
        weight_key,
        activation_key,
        activation_format: FusedMoEActivationFormat,
    ):
        if activation_key is not None:
            return False, "expects bf16 activations"
        return True, None


def test_nvfp4_auto_selection_never_probes_flashinfer_b12x(monkeypatch):
    """B12x stays explicit opt-in even if a fake implementation supports it."""

    probed: list[nvfp4.NvFp4MoeBackend] = []

    def fake_backend_to_kernel_cls(backend):
        probed.append(backend)
        return [_AlwaysSupportedExperts]

    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP4", raising=False)
    monkeypatch.delenv("VLLM_FLASHINFER_MOE_BACKEND", raising=False)
    monkeypatch.setattr(nvfp4, "backend_to_kernel_cls", fake_backend_to_kernel_cls)

    backend, experts_cls = nvfp4.select_nvfp4_moe_backend(
        _make_moe_config(),
        weight_key=kNvfp4Static,
        activation_key=kNvfp4Dynamic,
    )

    assert backend != nvfp4.NvFp4MoeBackend.FLASHINFER_B12X
    assert nvfp4.NvFp4MoeBackend.FLASHINFER_B12X not in probed
    assert experts_cls is _AlwaysSupportedExperts


def test_nvfp4_explicit_flashinfer_b12x_remains_available(monkeypatch):
    """An explicit ``--moe-backend flashinfer_b12x`` still maps to B12x."""

    probed: list[nvfp4.NvFp4MoeBackend] = []

    def fake_backend_to_kernel_cls(backend):
        probed.append(backend)
        return [_AlwaysSupportedExperts]

    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP4", raising=False)
    monkeypatch.delenv("VLLM_FLASHINFER_MOE_BACKEND", raising=False)
    monkeypatch.setattr(nvfp4, "backend_to_kernel_cls", fake_backend_to_kernel_cls)

    backend, experts_cls = nvfp4.select_nvfp4_moe_backend(
        _make_moe_config(moe_backend="flashinfer_b12x"),
        weight_key=kNvfp4Static,
        activation_key=None,
    )

    assert backend == nvfp4.NvFp4MoeBackend.FLASHINFER_B12X
    assert probed == [nvfp4.NvFp4MoeBackend.FLASHINFER_B12X]
    assert experts_cls is _AlwaysSupportedExperts


def test_nvfp4_w4a16_auto_selection_falls_back_to_marlin(monkeypatch):
    """W4A16 checkpoints use activation_key=None and must not hit W4A4 FI."""

    def fake_backend_to_kernel_cls(backend):
        if backend == nvfp4.NvFp4MoeBackend.MARLIN:
            return [_W4A16Experts]
        return [_W4A4OnlyExperts]

    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP4", raising=False)
    monkeypatch.delenv("VLLM_FLASHINFER_MOE_BACKEND", raising=False)
    monkeypatch.setattr(nvfp4, "backend_to_kernel_cls", fake_backend_to_kernel_cls)

    backend, experts_cls = nvfp4.select_nvfp4_moe_backend(
        _make_moe_config(),
        weight_key=kNvfp4Static,
        activation_key=None,
    )

    assert backend == nvfp4.NvFp4MoeBackend.MARLIN
    assert experts_cls is _W4A16Experts
