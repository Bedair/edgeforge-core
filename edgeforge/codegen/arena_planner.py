"""
EdgeForge — Arena Planner
Computes the TFLite Micro tensor arena size for a given model and MCU target.

The arena holds:
  - All activation tensors (intermediate results)
  - TFLite Micro internal bookkeeping (~4 KB overhead)
  - Alignment padding

For quantised (INT8) models, activations are 4x smaller than float32.
We add a 20% safety margin on top of the computed size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edgeforge.targets.loader import TargetProfile


# TFLite Micro internal overhead in bytes
_TFLITE_MICRO_OVERHEAD = 4096   # 4 KB bookkeeping
_SAFETY_MARGIN         = 1.20   # 20% headroom above computed minimum

# Default arena alignment (bytes) — 8 for Cortex-M4, 16 for M7/M33 with cache
_DEFAULT_ALIGNMENT = 8


@dataclass
class ArenaConfig:
    # Computed sizes
    activation_bytes:  int       # peak activation tensor bytes
    overhead_bytes:    int       # TFLite Micro bookkeeping
    padding_bytes:     int       # alignment padding
    total_bytes:       int       # final arena size to allocate
    total_kb:          float

    # Target info
    alignment:         int       # byte alignment required
    ccm_eligible:      bool      # can use CCM SRAM (STM32 only)
    ccm_size_kb:       int       # available CCM KB (0 if not applicable)

    # Headroom
    ram_headroom_bytes: int
    ram_headroom_kb:    float
    scratch_bytes:      int = 2048  # CMSIS-NN scratch buffer

    @property
    def total_bytes_aligned(self) -> int:
        """Arena size rounded up to alignment boundary."""
        a = self.alignment
        return ((self.total_bytes + a - 1) // a) * a

    @property
    def c_define_value(self) -> str:
        """Value for the EDGEFORGE_ARENA_SIZE C define."""
        return str(self.total_bytes_aligned)

    @property
    def fits_in_ccm(self) -> bool:
        """True if arena fits entirely in CCM SRAM."""
        return self.ccm_eligible and self.total_kb <= self.ccm_size_kb


def plan_arena(
    model_path: str | Path,
    target:     "TargetProfile",
) -> ArenaConfig:
    """
    Compute the TFLite Micro arena configuration for the given model and target.

    Args:
        model_path: Path to the optimised .onnx file.
        target:     MCU target profile.

    Returns:
        ArenaConfig with all sizing information.
    """
    p = Path(model_path)
    activation_bytes = _estimate_peak_activation(p)

    # Apply safety margin and overhead
    with_margin   = int(activation_bytes * _SAFETY_MARGIN)
    total_bytes   = with_margin + _TFLITE_MICRO_OVERHEAD

    # Alignment — Cortex-M7/M33 with cache needs 16-byte alignment
    alignment = _get_alignment(target)

    # Align up
    total_bytes = _align_up(total_bytes, alignment)
    padding     = total_bytes - with_margin - _TFLITE_MICRO_OVERHEAD

    # CCM SRAM eligibility (STM32F4/F7 family)
    ccm_kb      = _get_ccm_kb(target)
    ccm_eligible = ccm_kb > 0

    # RAM headroom after arena
    ram_headroom_bytes = (target.ram_kb * 1024) - total_bytes
    ram_headroom_kb    = ram_headroom_bytes / 1024

    return ArenaConfig(
        activation_bytes=activation_bytes,
        overhead_bytes=_TFLITE_MICRO_OVERHEAD,
        padding_bytes=padding,
        total_bytes=total_bytes,
        total_kb=total_bytes / 1024,
        alignment=alignment,
        ccm_eligible=ccm_eligible,
        ccm_size_kb=ccm_kb,
        ram_headroom_bytes=max(ram_headroom_bytes, 0),
        ram_headroom_kb=max(ram_headroom_kb, 0.0),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimate_peak_activation(model_path: Path) -> int:
    """
    Estimate peak activation memory from ONNX graph.
    For INT8 models, activations are 1 byte/element.
    For float32 models, 4 bytes/element.
    """
    try:
        import onnx

        model = onnx.load(str(model_path))
        graph = model.graph

        # Detect quantisation
        is_q = any(
            n.op_type in {"DequantizeLinear", "QuantizeLinear",
                          "ConvInteger", "MatMulInteger"}
            for n in graph.node
        )
        bytes_per = 1 if is_q else 4

        # Find peak activation tensor
        max_act = 0
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            t = vi.type.tensor_type
            if not t.HasField("elem_type"):
                continue
            shape = [max(d.dim_value, 1) for d in t.shape.dim]
            if not shape:
                continue
            size = math.prod(shape) * bytes_per
            max_act = max(max_act, size)

        # Minimum floor — even tiny models need some workspace
        return max(max_act, 8192)

    except Exception:
        # Safe fallback: 64 KB
        return 65536


def _get_alignment(target: "TargetProfile") -> int:
    """Return required arena byte alignment for the target core."""
    core = target.core.lower()
    if "m7" in core or "m33" in core or "m55" in core or "m85" in core:
        return 16   # cache line alignment for M7/M33+
    return 8        # standard for M0/M4


def _get_ccm_kb(target: "TargetProfile") -> int:
    """Return CCM SRAM size in KB if applicable, else 0."""
    # STM32F4: 64 KB CCM, STM32F7: 128 KB DTCM
    tid = target.id.lower()
    if "stm32f4" in tid or "stm32f407" in tid:
        return 64
    if "stm32f7" in tid:
        return 128
    if "stm32h7" in tid:
        return 128
    return 0


def _align_up(value: int, alignment: int) -> int:
    """Round value up to the next multiple of alignment."""
    return ((value + alignment - 1) // alignment) * alignment
