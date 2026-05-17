"""EdgeForge CLI — main entry point."""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Fix Windows Unicode encoding for terminals that don't support UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import click
from rich.console import Console
from rich.table   import Table
from rich         import box

# Force rich to use ASCII-safe mode on Windows legacy terminals
_FORCE_TERMINAL = sys.platform != "win32" or os.environ.get("WT_SESSION") or os.environ.get("TERM")
console = Console(highlight=False, emoji=False)

# ASCII-safe status icons (work on all Windows terminals)
OK   = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


@click.group()
@click.version_option(package_name="edgeforge")
def main():
    """EdgeForge -- forge your models into firmware."""
    pass


# ── edgeforge analyze ────────────────────────────────────────────────────────

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mcu",      default=None, help="Filter to a specific MCU.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def analyze(model_path: str, mcu: str | None, as_json: bool):
    """Analyze a model -- format, graph, RAM/flash estimate, board compatibility."""
    from edgeforge.converter.detector import detect, describe, ModelFormat
    from edgeforge.converter.to_onnx  import to_onnx, ConversionError
    from edgeforge.converter.analyzer import analyze as _analyze
    from edgeforge.targets.loader     import (
        load_target, check_compatibility, check_all_targets, FitStatus,
    )

    p = Path(model_path)
    console.rule("EdgeForge Analyze")

    with console.status(f"Detecting format of {p.name}..."):
        info = describe(p)
        fmt  = info["format"]

    console.print(
        f"\nModel:  {p.name}  ({info['size_human']})\n"
        f"Format: {fmt.value.upper()}"
    )

    if fmt == ModelFormat.UNKNOWN:
        console.print(f"{FAIL} Cannot detect format. Supported: .tflite .onnx .pt .pb SavedModel")
        sys.exit(1)

    if fmt == ModelFormat.ONNX:
        onnx_path = p
        console.print(f"{OK} Already ONNX -- skipping conversion.")
    else:
        with console.status("Converting to ONNX IR..."):
            try:
                onnx_path = to_onnx(p)
                console.print(f"{OK} Converted to ONNX IR")
            except ConversionError as e:
                console.print(f"{FAIL} Conversion failed: {e}")
                sys.exit(1)

    with console.status("Analyzing model graph..."):
        try:
            result = _analyze(onnx_path, original_format=fmt.value)
        except Exception as e:
            console.print(f"{FAIL} Analysis failed: {e}")
            sys.exit(1)

    console.print()
    console.rule("Graph Summary")
    op_str = "  ".join(
        f"{op} x{count}"
        for op, count in sorted(result.op_counts.items(), key=lambda x: -x[1])
    )
    console.print(f"Operators:  {result.total_ops} total\n  {op_str}")
    console.print(f"Parameters: {result.parameter_count:,}")

    console.print()
    for ti in result.input_tensors:
        console.print(
            f"Input:  {ti.name}  "
            f"({' x '.join(str(d) for d in ti.shape)}  {ti.dtype})"
        )
    for ti in result.output_tensors:
        console.print(
            f"Output: {ti.name}  "
            f"({' x '.join(str(d) for d in ti.shape)}  {ti.dtype})"
        )

    console.print()
    console.rule("Memory Estimates")
    console.print(
        f"Flash (INT8):  {result.flash_kb:.1f} KB\n"
        f"RAM   (arena): {result.arena_kb:.1f} KB"
    )

    console.print()
    console.rule("Board Compatibility")

    if mcu:
        try:
            t = load_target(mcu)
            compat_list = [check_compatibility(result.arena_kb, result.flash_kb, t)]
        except FileNotFoundError as e:
            console.print(f"{FAIL} {e}")
            sys.exit(1)
    else:
        compat_list = check_all_targets(result.arena_kb, result.flash_kb)

    table = Table(box=box.SIMPLE, header_style="bold dim")
    table.add_column("Board");  table.add_column("Core", style="dim")
    table.add_column("RAM",  justify="right"); table.add_column("Flash", justify="right")
    table.add_column("Arena", justify="right"); table.add_column("Status", justify="center")

    _icons = {
        "fits":      "FITS",
        "tight":     "TIGHT",
        "too_small": "TOO SMALL",
    }

    for c in compat_list:
        t = c.target
        worst = (
            "too_small" if (c.ram_status == FitStatus.TOOSMALL or c.flash_status == FitStatus.TOOSMALL)
            else "tight" if (c.ram_status == FitStatus.TIGHT or c.flash_status == FitStatus.TIGHT)
            else "fits"
        )
        table.add_row(
            t.name, t.core,
            f"{t.ram_kb} KB", f"{t.flash_kb} KB",
            f"{c.arena_kb:.0f} KB",
            _icons[worst],
        )
        for w in c.warnings:
            table.add_row("", "", "", "", "", f"  {WARN} {w}")
    console.print(table)

    if as_json:
        import json
        out = {
            "model": str(p), "format": fmt.value,
            "parameter_count": result.parameter_count,
            "flash_kb": result.flash_kb, "arena_kb": result.arena_kb,
            "compatibility": [
                {"target_id": c.target.id, "fits": c.fits, "warnings": c.warnings}
                for c in compat_list
            ],
        }
        console.print_json(json.dumps(out, indent=2))


# ── edgeforge optimize ───────────────────────────────────────────────────────

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mcu",        required=True, help="Target MCU profile ID.")
@click.option("--output", "-o", default=None, help="Output .onnx path.")
@click.option(
    "--mode", "quant_mode", default="dynamic",
    type=click.Choice(["dynamic", "static"]),
    help="Quantisation mode.",
)
@click.option("--calibration-dir", default=None,
              help="Directory of .npy files for static quantisation.")
def optimize(model_path: str, mcu: str, output: str | None,
             quant_mode: str, calibration_dir: str | None):
    """Optimise a model to fit the target MCU -- quantise, simplify, check budget."""
    from edgeforge.optimizer.optimizer import optimize as _optimize, OptimizeError
    from edgeforge.targets.loader      import load_target
    from edgeforge.optimizer.budget    import format_bar

    p = Path(model_path)
    console.rule("EdgeForge Optimize")

    try:
        target = load_target(mcu)
    except FileNotFoundError as e:
        console.print(f"{FAIL} {e}"); sys.exit(1)

    console.print(
        f"\nModel:  {p.name}\n"
        f"Target: {target.name}  (RAM {target.ram_kb} KB  Flash {target.flash_kb} KB)\n"
        f"Mode:   {quant_mode} quantisation\n"
    )

    out_path = Path(output) if output else None

    with console.status("Running optimisation pipeline..."):
        try:
            result = _optimize(
                src=p, target=target, output_path=out_path,
                mode=quant_mode,
                calibration_dir=calibration_dir,
                strict=False,
            )
        except OptimizeError as e:
            console.print(f"{FAIL} Optimisation failed: {e}"); sys.exit(1)
        except Exception as e:
            console.print(f"{FAIL} Unexpected error: {e}"); sys.exit(1)

    console.rule("Steps Applied")
    for step in result.steps_applied:
        console.print(f"  {OK} {step}")
    sr = result.simplify_report
    if sr.nodes_saved > 0:
        console.print(
            f"  Graph: {sr.nodes_before} -> {sr.nodes_after} nodes "
            f"(-{sr.nodes_saved} removed/fused)"
        )
    qr = result.quantize_report
    if qr.already_quantized:
        console.print(
            f"  [NOTE] Model is already INT8 quantised -- no further "
            f"size reduction was applied."
        )
        console.print(
            f"  [NOTE] Flash size reflects the quantised weights as-is. "
            f"If it still exceeds flash budget, the model architecture "
            f"itself is too large for this MCU. Consider a smaller variant."
        )

    console.print()
    console.rule("Size Reduction")
    delta = result.flash_before_kb - result.flash_after_kb
    sign  = "-" if delta >= 0 else "+"
    label = "saved" if delta >= 0 else "grew"
    console.print(
        f"  Flash  {result.flash_before_kb:>8.1f} KB  ->  "
        f"{result.flash_after_kb:>8.1f} KB  "
        f"({sign}{abs(result.flash_reduction_pct):.0f}% {label})"
    )
    if result.quantize_report.already_quantized:
        console.print(
            f"  [NOTE] Model is already INT8 -- no quantisation applied."
        )
        console.print(
            f"  [NOTE] To fit this MCU, a smaller architecture is needed"
            f" (e.g. DS-CNN-S for audio, MobileNetV2 0.35x for vision)."
        )

    console.print()
    console.rule(f"Budget Check -- {target.name}")
    br = result.budget_report
    ram_icon   = OK   if br.ram_fits   else FAIL
    flash_icon = OK   if br.flash_fits else FAIL
    console.print(
        f"  RAM    {format_bar(br.ram_used_pct)}  "
        f"{br.arena_kb:>6.0f} / {br.target_ram_kb} KB  {ram_icon}"
    )
    console.print(
        f"  Flash  {format_bar(br.flash_used_pct)}  "
        f"{br.flash_kb:>6.0f} / {br.target_flash_kb} KB  {flash_icon}"
    )
    if br.suggestions:
        console.print()
        for tip in br.suggestions:
            console.print(f"  {WARN}  {tip}")

    console.print()
    if result.fits:
        console.print(f"{OK} Fits {target.id}  ->  {result.output_path}")
    else:
        console.print(
            f"{WARN} Written but does not fit {target.id}  ->  {result.output_path}"
        )


# ── edgeforge compile ────────────────────────────────────────────────────────

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mcu", required=True, help="Target MCU profile ID.")
@click.option("--rtos", default="none",
              type=click.Choice(["none", "freertos", "zephyr"]),
              help="RTOS for generated glue code.")
@click.option("--output-dir", "-o", default="edgeforge_output",
              help="Output directory for generated files.")
def compile(model_path: str, mcu: str, rtos: str, output_dir: str):
    """Compile a model to C/C++ files ready for your firmware project."""
    from edgeforge.targets.loader import load_target
    from edgeforge.codegen.codegen import generate, CodegenError

    p = Path(model_path)
    console.rule("EdgeForge Compile")

    try:
        target = load_target(mcu)
    except FileNotFoundError as e:
        console.print(f"{FAIL} {e}"); sys.exit(1)

    console.print(f"Model:  {p.name}")
    console.print(f"Target: {target.name}")
    console.print(f"RTOS:   {rtos}")
    console.print(f"Output: {output_dir}")

    with console.status("Generating C/C++ files..."):
        try:
            result = generate(
                model_path=p,
                target=target,
                rtos=rtos,
                output_dir=output_dir,
            )
        except (CodegenError, FileNotFoundError) as e:
            console.print(f"{FAIL} Code generation failed: {e}"); sys.exit(1)
        except Exception as e:
            console.print(f"{FAIL} Unexpected error: {e}"); sys.exit(1)

    console.rule("Generated Files")
    for f in result.files_written:
        console.print(f"  {OK} {f}")

    console.print()
    mi = result.model_info
    ar = result.arena_config
    console.print(f"Model:  {mi.node_count} nodes  {mi.op_summary[:60]}")
    console.print(f"Arena:  {ar.total_bytes_aligned} bytes ({ar.total_kb:.1f} KB)")
    console.print(f"RAM left after arena:  {ar.ram_headroom_kb:.1f} KB")
    if ar.ccm_eligible and ar.fits_in_ccm:
        console.print(f"[NOTE] Arena fits in CCM SRAM -- set EDGEFORGE_USE_CCM=1 for better performance")

    console.print()
    console.print(f"{OK} Output written to: {result.output_dir}")


# ── edgeforge targets ────────────────────────────────────────────────────────

@main.command("targets")
@click.option("--mcu", default=None)
def list_targets(mcu: str | None):
    """List all supported MCU targets."""
    from edgeforge.targets.loader import load_target, all_targets

    if mcu:
        try:
            t = load_target(mcu)
        except FileNotFoundError as e:
            console.print(f"{FAIL} {e}"); sys.exit(1)
        console.print()
        console.rule(t.name)
        console.print(f"  ID: {t.id}  Vendor: {t.vendor}")
        console.print(f"  Core: {t.core}  FPU: {t.fpu}  NPU: {t.npu}")
        console.print(f"  RAM: {t.ram_kb} KB  Flash: {t.flash_kb} KB")
        console.print(f"  CMSIS-NN: {t.cmsis_nn}  Runtime: {t.runtime}")
        rtos = ", ".join(
            r for r, ok in [("FreeRTOS", t.rtos_freertos), ("Zephyr", t.rtos_zephyr)] if ok
        ) or "--"
        console.print(f"  RTOS: {rtos}")
        console.print(f"  Flags: {t.compiler_flags}")
        return

    table = Table(box=box.SIMPLE, header_style="bold dim")
    table.add_column("ID", style="bold"); table.add_column("Name")
    table.add_column("Core", style="dim")
    table.add_column("RAM",  justify="right"); table.add_column("Flash", justify="right")
    table.add_column("NPU",  justify="center"); table.add_column("RTOS", style="dim")

    for t in all_targets():
        rtos = ", ".join(
            r for r, ok in [("FreeRTOS", t.rtos_freertos), ("Zephyr", t.rtos_zephyr)] if ok
        ) or "--"
        table.add_row(
            t.id, t.name, t.core,
            f"{t.ram_kb} KB", f"{t.flash_kb} KB",
            "YES" if t.npu else "--",
            rtos,
        )
    console.print()
    console.rule("EdgeForge Supported Targets")
    console.print(table)


if __name__ == "__main__":
    main()
