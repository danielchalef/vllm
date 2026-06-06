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


def _make_scale_tensors():
    w13_scale = torch.ones((2, 4, 8), dtype=torch.float32)
    w2_scale = torch.ones((2, 4, 8), dtype=torch.float32) * 2
    w13_scale_2 = torch.tensor([0.5, 0.25], dtype=torch.float32)
    w2_scale_2 = torch.tensor([0.125, 0.0625], dtype=torch.float32)
    a13_scale = torch.tensor([0.25, 0.5], dtype=torch.float32)
    a2_scale = torch.tensor([0.125, 0.25], dtype=torch.float32)
    return w13_scale, w2_scale, w13_scale_2, w2_scale_2, a13_scale, a2_scale


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


def test_nvfp4_w4a16_quant_config_suppresses_activation_scales():
    (
        w13_scale,
        w2_scale,
        w13_scale_2,
        w2_scale_2,
        _a13_scale,
        _a2_scale,
    ) = _make_scale_tensors()

    quant_config = nvfp4.make_nvfp4_moe_quant_config(
        backend=nvfp4.NvFp4MoeBackend.FLASHINFER_B12X,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=None,
        a2_scale=None,
        source_format="modelopt",
    )

    assert quant_config.use_nvfp4_w4a16
    assert quant_config.quant_dtype is None
    assert quant_config.weight_quant_dtype == "nvfp4"
    assert quant_config.a1_gscale is None
    assert quant_config.a2_gscale is None
    assert quant_config.g1_alphas is w13_scale_2
    assert quant_config.g2_alphas is w2_scale_2
    assert quant_config.source_format == "modelopt"


def test_nvfp4_w4a4_quant_config_preserves_activation_scales():
    (
        w13_scale,
        w2_scale,
        w13_scale_2,
        w2_scale_2,
        a13_scale,
        a2_scale,
    ) = _make_scale_tensors()

    quant_config = nvfp4.make_nvfp4_moe_quant_config(
        backend=nvfp4.NvFp4MoeBackend.FLASHINFER_B12X,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
        source_format="compressed_tensors",
    )

    assert quant_config.use_nvfp4_w4a4
    assert quant_config.weight_quant_dtype == "nvfp4"
    torch.testing.assert_close(quant_config.a1_gscale, 1.0 / a13_scale)
    torch.testing.assert_close(quant_config.a2_gscale, 1.0 / a2_scale)
    assert quant_config.source_format == "compressed_tensors"


def test_flashinfer_b12x_tracks_activation_precision_and_source_format():
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
        FlashInferB12xExperts,
    )

    (
        w13_scale,
        w2_scale,
        w13_scale_2,
        w2_scale_2,
        a13_scale,
        a2_scale,
    ) = _make_scale_tensors()

    w4a16_quant_config = nvfp4.make_nvfp4_moe_quant_config(
        backend=nvfp4.NvFp4MoeBackend.FLASHINFER_B12X,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=None,
        a2_scale=None,
        source_format="compressed_tensors",
    )
    w4a16_experts = FlashInferB12xExperts(
        moe_config=_make_moe_config(moe_backend="flashinfer_b12x"),
        quant_config=w4a16_quant_config,
    )
    assert w4a16_experts.activation_precision == "bf16"
    assert w4a16_experts.source_format == "compressed_tensors"

    w4a4_quant_config = nvfp4.make_nvfp4_moe_quant_config(
        backend=nvfp4.NvFp4MoeBackend.FLASHINFER_B12X,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
    )
    w4a4_experts = FlashInferB12xExperts(
        moe_config=_make_moe_config(moe_backend="flashinfer_b12x"),
        quant_config=w4a4_quant_config,
    )
    assert w4a4_experts.activation_precision == "fp4"
    assert w4a4_experts.source_format == "modelopt"


def test_compressed_tensors_w4a16_moe_suppresses_activation_scales(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_w4a4_nvfp4 as ct_nvfp4_moe,
    )

    calls = []

    def fake_select_nvfp4_moe_backend(**kwargs):
        calls.append(kwargs)
        return nvfp4.NvFp4MoeBackend.FLASHINFER_B12X, _AlwaysSupportedExperts

    monkeypatch.setattr(
        ct_nvfp4_moe,
        "select_nvfp4_moe_backend",
        fake_select_nvfp4_moe_backend,
    )
    monkeypatch.setattr(
        ct_nvfp4_moe,
        "is_global_sf_supported_for_nvfp4_backend",
        lambda _backend: False,
    )

    method = ct_nvfp4_moe.CompressedTensorsW4A4Nvfp4MoEMethod(
        _make_moe_config(),
        use_a16=True,
    )

    assert method.use_a16 is True
    assert calls[0]["weight_key"] is kNvfp4Static
    assert calls[0]["activation_key"] is None

    (
        w13_scale,
        w2_scale,
        w13_scale_2,
        w2_scale_2,
        a13_scale,
        a2_scale,
    ) = _make_scale_tensors()
    layer = torch.nn.Module()
    layer.w13_weight_scale = w13_scale
    layer.w2_weight_scale = w2_scale
    layer.w13_weight_scale_2 = w13_scale_2
    layer.w2_weight_scale_2 = w2_scale_2
    layer.w13_input_scale = a13_scale
    layer.w2_input_scale = a2_scale

    quant_config = method.get_fused_moe_quant_config(layer)

    assert quant_config.use_nvfp4_w4a16
    assert quant_config.a1_gscale is None
    assert quant_config.a2_gscale is None
    assert quant_config.source_format == "compressed_tensors"
