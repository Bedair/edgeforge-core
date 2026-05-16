"""
EdgeForge — Optimiser Orchestrator
Single entry point for the full Phase 2 optimisation pipeline:

  ONNX IR → simplify → quantise → budget check → output

Combines simplifier, quantizer, and budget checker into one call.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import onnx

if TYPE_CHECKING:
    from edgeforge.targets.loader import TargetProfile

from .simplifier import simplify, SimplifyReport
from .quantizer  import quantize_dynamic, quantize_static, QuantizeReport
from .budget     import check_budget, BudgetReport

log = logging.getLogger(__name__)


@dataclass
class OptimizeResult:
    # Paths
    input_path:  str
    output_path: str

    # Stats
    flash_before_kb:  float
    flash_after_kb:   float
    flash_reduction_pct: float

    # Sub-reports
    simplify_report:  SimplifyReport
    quantize_report:  QuantizeReport
    budget_report:    BudgetReport

    # Applied steps
    steps_applied: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.budget_report.fits

    @property
    def success(self) -> bool:
        return Path(self.output_path).exists()


class OptimizeError(Exception):
    """Raised when optimisation cannot produce a model that fits the target."""
    pass


def optimize(
    src:             str | Path,
    target:          "TargetProfile",
    output_path:     str | Path | None = None,
    mode:            str = "dynamic",
    calibration_data: list | None = None,
    calibration_dir: str | Path | None = None,
    strict:          bool = False,
) -> OptimizeResult:
    """
    Run the full EdgeForge optimisation pipeline on a model.

    Pipeline:
      1. Convert to ONNX IR (if not already)
      2. Simplify (constant folding, dead node removal, BN fusion)
      3. Quantise (dynamic INT8 by default, static if calibration data given)
      4. Check MCU budget
      5. Write output

    Args:
        src:              Path to input model (.onnx, .tflite, .pt, .pb).
        target:           MCU target profile (from targets/loader.py).
        output_path:      Where to write the optimised .onnx.
                          Defaults to <src_stem>_opt.onnx next to src.
        mode:             "dynamic" or "static" quantisation.
        calibration_data: For static mode — list of numpy arrays.
        calibration_dir:  For static mode — directory of .npy files.
        strict:           If True, raise OptimizeError when model doesn't fit.
                          If False (default), warn and return anyway.

    Returns:
        OptimizeResult with full stats.

    Raises:
        OptimizeError: If strict=True and model doesn't fit the target.
        FileNotFoundError: If src doesn't exist.
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Input model not found: {src}")

    # Resolve output path
    if output_path is None:
        output_path = src.parent / (src.stem + "_opt.onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flash_before_kb = src.stat().st_size / 1024
    steps: list[str] = []

    with tempfile.TemporaryDirectory(prefix="edgeforge_opt_") as tmp:
        tmp_path = Path(tmp)

        # ── Step 1: Ensure ONNX IR ───────────────────────────────────────────
        from edgeforge.converter.detector import detect, ModelFormat
        from edgeforge.converter.to_onnx  import to_onnx, ConversionError

        fmt = detect(src)
        if fmt == ModelFormat.ONNX:
            onnx_path = src
        else:
            log.info(f"Converting {fmt.value} → ONNX IR")
            try:
                onnx_path = to_onnx(src, dst=tmp_path / "input.onnx")
                steps.append(f"convert({fmt.value}→onnx)")
            except ConversionError as e:
                raise OptimizeError(f"Conversion failed: {e}") from e

        # ── Step 2: Simplify ─────────────────────────────────────────────────
        simplified_path = tmp_path / "simplified.onnx"
        try:
            model = onnx.load(str(onnx_path))
            model_simplified, simplify_report = simplify(model)
            onnx.save(model_simplified, str(simplified_path))
            if simplify_report.nodes_saved > 0:
                steps.append(
                    f"simplify(-{simplify_report.nodes_saved} nodes)"
                )
            log.info(f"Simplify: {simplify_report.passes_applied}")
        except Exception as e:
            log.warning(f"Simplification failed ({e}), skipping")
            shutil.copy2(onnx_path, simplified_path)
            simplify_report = SimplifyReport(
                nodes_before=0, nodes_after=0,
                passes_applied=["skipped"],
            )

        # ── Step 3: Quantise ─────────────────────────────────────────────────
        quantized_path = tmp_path / "quantized.onnx"

        # Check if already quantised before attempting
        from .quantizer import is_already_quantized
        if is_already_quantized(simplified_path):
            log.info("Model is already quantised -- skipping quantisation step")
            import shutil as _shutil
            _shutil.copy2(simplified_path, quantized_path)
            from .quantizer import QuantizeReport
            quant_report = QuantizeReport(
                mode="skipped",
                flash_before_kb=simplified_path.stat().st_size / 1024,
                flash_after_kb=simplified_path.stat().st_size / 1024,
                flash_reduction_pct=0.0,
                nodes_quantized=0,
                accuracy_delta_est="model already INT8 quantised",
                already_quantized=True,
            )
            steps.append("quantize_skipped(already_int8)")
        elif mode == "static" and (calibration_data or calibration_dir):
            log.info("Applying static INT8 quantisation")
            try:
                quant_report = quantize_static(
                    simplified_path, quantized_path,
                    calibration_data=calibration_data,
                    calibration_dir=calibration_dir,
                )
                steps.append("quantize_static_int8")
            except Exception as e:
                log.warning(f"Static quant failed ({e}), falling back to dynamic")
                quant_report = quantize_dynamic(simplified_path, quantized_path)
                steps.append("quantize_dynamic_int8(fallback)")
        else:
            log.info("Applying dynamic INT8 quantisation")
            quant_report = quantize_dynamic(simplified_path, quantized_path)
            steps.append("quantize_dynamic_int8")

        # ── Step 4: Copy to output ───────────────────────────────────────────
        shutil.copy2(quantized_path, output_path)

        # ── Step 5: Budget check ─────────────────────────────────────────────
        budget_report = check_budget(output_path, target)

        flash_after_kb      = output_path.stat().st_size / 1024
        flash_reduction_pct = (
            (1 - flash_after_kb / flash_before_kb) * 100
            if flash_before_kb > 0 else 0.0
        )

        result = OptimizeResult(
            input_path=str(src),
            output_path=str(output_path),
            flash_before_kb=flash_before_kb,
            flash_after_kb=flash_after_kb,
            flash_reduction_pct=flash_reduction_pct,
            simplify_report=simplify_report,
            quantize_report=quant_report,
            budget_report=budget_report,
            steps_applied=steps,
        )

        if strict and not budget_report.fits:
            suggestions = "\n  ".join(budget_report.suggestions)
            raise OptimizeError(
                f"Optimised model does not fit {target.id}.\n"
                f"  RAM:   {budget_report.arena_kb:.0f} KB / {target.ram_kb} KB\n"
                f"  Flash: {budget_report.flash_kb:.0f} KB / {target.flash_kb} KB\n"
                f"Suggestions:\n  {suggestions}"
            )

        return result
