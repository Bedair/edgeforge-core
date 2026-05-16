"""
EdgeForge — MCU Budget Checker
Validates whether an optimised model fits within a target MCU's
RAM and flash constraints, and suggests next steps if it doesn't.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edgeforge.targets.loader import TargetProfile


@dataclass
class BudgetReport:
    # Model stats
    flash_kb:      float
    arena_kb:      float

    # Target stats
    target_id:     str
    target_ram_kb: int
    target_flash_kb: int

    # Fit result
    ram_fits:      bool
    flash_fits:    bool

    # Headroom (how much is left after the model)
    ram_headroom_kb:   float
    flash_headroom_kb: float
    ram_used_pct:      float
    flash_used_pct:    float

    # Advice
    suggestions: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.ram_fits and self.flash_fits

    @property
    def status(self) -> str:
        if self.fits:
            if self.ram_used_pct > 80 or self.flash_used_pct > 60:
                return "tight"
            return "ok"
        return "fail"


# Flash safety margin — leave 40% of flash for firmware code
_FLASH_MODEL_BUDGET_PCT = 0.60

# RAM tight threshold — warn when arena uses > 70% of RAM
_RAM_TIGHT_PCT = 70.0


def check_budget(
    model_path: "str | Path",
    target: "TargetProfile",
) -> BudgetReport:
    """
    Check whether a model file fits within the target MCU's memory budget.

    Computes:
    - flash_kb  = model file size (weight storage, INT8 assumed)
    - arena_kb  = estimated TFLite Micro inference arena

    Args:
        model_path: Path to the (optimised) .onnx model file.
        target:     MCU target profile from the loader.

    Returns:
        BudgetReport dataclass.
    """
    p = Path(model_path)
    flash_kb = p.stat().st_size / 1024
    arena_kb = _estimate_arena(p)

    flash_budget_kb = target.flash_kb * _FLASH_MODEL_BUDGET_PCT
    ram_fits        = arena_kb  <= target.ram_kb
    flash_fits      = flash_kb  <= flash_budget_kb

    ram_headroom_kb   = target.ram_kb   - arena_kb
    flash_headroom_kb = flash_budget_kb - flash_kb
    ram_used_pct      = (arena_kb  / target.ram_kb)   * 100
    flash_used_pct    = (flash_kb  / target.flash_kb) * 100

    suggestions = suggest_strategy(
        flash_kb=flash_kb,
        arena_kb=arena_kb,
        target=target,
        ram_fits=ram_fits,
        flash_fits=flash_fits,
        ram_used_pct=ram_used_pct,
    )

    return BudgetReport(
        flash_kb=flash_kb,
        arena_kb=arena_kb,
        target_id=target.id,
        target_ram_kb=target.ram_kb,
        target_flash_kb=target.flash_kb,
        ram_fits=ram_fits,
        flash_fits=flash_fits,
        ram_headroom_kb=ram_headroom_kb,
        flash_headroom_kb=flash_headroom_kb,
        ram_used_pct=ram_used_pct,
        flash_used_pct=flash_used_pct,
        suggestions=suggestions,
    )


def suggest_strategy(
    flash_kb:   float,
    arena_kb:   float,
    target:     "TargetProfile",
    ram_fits:   bool,
    flash_fits: bool,
    ram_used_pct: float,
) -> list[str]:
    """
    Generate actionable suggestions based on memory fit status.

    Returns a list of human-readable suggestion strings.
    """
    tips: list[str] = []

    if not flash_fits:
        tips.append(
            f"Flash too large ({flash_kb:.0f} KB > "
            f"{target.flash_kb * _FLASH_MODEL_BUDGET_PCT:.0f} KB budget). "
            "Apply INT8 dynamic quantisation to reduce weights by ~75%."
        )
        tips.append(
            "If already quantised, consider a smaller architecture: "
            "MobileNetV2 0.35x or EfficientNet-Lite0 for vision tasks, "
            "DS-CNN-S for audio."
        )

    if not ram_fits:
        tips.append(
            f"Arena too large ({arena_kb:.0f} KB > {target.ram_kb} KB RAM). "
            "Apply static INT8 quantisation to reduce activation memory."
        )
        tips.append(
            "Enable CMSIS-NN on this target to use in-place operator "
            "optimisations that reduce peak activation memory."
        )
        tips.append(
            f"Consider a larger target: PSoC 6 (288 KB RAM) or "
            f"nRF52840 (256 KB RAM) if STM32F407 (192 KB) is too small."
        )

    if ram_fits and flash_fits:
        if ram_used_pct > _RAM_TIGHT_PCT:
            tips.append(
                f"RAM is tight ({ram_used_pct:.0f}% used). "
                "Leave headroom for sensor buffers and RTOS stack. "
                "Apply static quantisation to reduce activation memory."
            )
        if not target.cmsis_nn:
            tips.append(
                "This target does not have CMSIS-NN support. "
                "Inference will be slow. Consider a Cortex-M4F target."
            )

    return tips


def _estimate_arena(model_path: Path) -> float:
    """
    Estimate TFLite Micro arena size from ONNX model.

    Strategy:
    - Find the largest intermediate tensor by byte count
    - Arena ≈ peak_activation × 1.5 + 4 KB bookkeeping overhead
    - For INT8 models, activation tensors are ~4× smaller than float32

    This is an estimate; exact sizing requires running tflite-micro's
    memory planner on the actual .tflite file.
    """
    try:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(model_path))
        graph = model.graph

        # Determine if model is already quantised (INT8)
        is_quantised = any(
            "QLinear" in n.op_type or "QuantizeLinear" in n.op_type
            for n in graph.node
        )
        bytes_per_elem = 1 if is_quantised else 4

        # Find peak activation tensor size
        max_activation = 0
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            t = vi.type.tensor_type
            if not t.HasField("elem_type"):
                continue
            shape = [max(d.dim_value, 1) for d in t.shape.dim]
            if not shape:
                continue
            size = math.prod(shape) * bytes_per_elem
            if size > max_activation:
                max_activation = size

        # Arena = peak activation × 1.5 (double-buffering) + 4 KB overhead
        arena_bytes = int(max_activation * 1.5) + 4096
        return arena_bytes / 1024

    except Exception:
        # Fallback: estimate from file size
        file_kb = model_path.stat().st_size / 1024
        return max(file_kb * 0.5, 16.0)


def format_bar(used_pct: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar for terminal output."""
    capped = min(used_pct, 100.0)
    filled = int(width * capped / 100)
    bar    = "#" * filled + "." * (width - filled)
    return bar
