"""
EdgeForge CLI — main entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.columns import Columns
from rich import print as rprint

console = Console()


@click.group()
@click.version_option(package_name="edgeforge")
def main():
    """
    ⚒  EdgeForge — forge your models into firmware.

    Takes any trained AI model and generates production-ready
    C/C++ code for your target MCU.
    """
    pass


# ── edgeforge analyze ────────────────────────────────────────────────────────

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--mcu", default=None,
    help="Filter compatibility report to a specific MCU (e.g. stm32f407). "
         "Omit to show all known targets."
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def analyze(model_path: str, mcu: str | None, as_json: bool):
    """
    Analyze a model — detect format, extract graph info, estimate RAM/flash,
    and show compatibility with your target MCU(s).

    MODEL_PATH can be a .tflite, .onnx, .pt, .pth, .pb file,
    or a TensorFlow SavedModel directory.
    """
    from .converter.detector import detect, describe, ModelFormat
    from .converter.to_onnx import to_onnx, ConversionError
    from .converter.analyzer import analyze as _analyze
    from .targets.loader import (
        load_target, check_compatibility, check_all_targets,
        FitStatus, all_targets,
    )

    p = Path(model_path)

    # ── Step 1: Detect format ────────────────────────────────────────────────
    console.rule("[bold]EdgeForge Analyze[/bold]")
    with console.status(f"Detecting format of [cyan]{p.name}[/cyan]..."):
        info = describe(p)
        fmt  = info["format"]

    _fmt_colors = {
        "tflite":      "green",
        "onnx":        "blue",
        "torchscript": "yellow",
        "tf_savedmodel": "magenta",
        "tf_frozen":   "magenta",
        "unknown":     "red",
    }
    color = _fmt_colors.get(fmt.value, "white")

    console.print(
        f"\n[bold]Model:[/bold]  {p.name}  "
        f"[dim]({info['size_human']})[/dim]\n"
        f"[bold]Format:[/bold] [{color}]{fmt.value.upper()}[/{color}]"
    )

    if fmt == ModelFormat.UNKNOWN:
        console.print(
            "[red]✗[/red] Could not detect model format. "
            "Ensure the file is a valid .tflite, .onnx, .pt, .pb, "
            "or SavedModel directory."
        )
        sys.exit(1)

    # ── Step 2: Convert to ONNX IR ───────────────────────────────────────────
    if fmt == ModelFormat.ONNX:
        onnx_path = p
        console.print("[green]✓[/green] Already in ONNX format — skipping conversion.")
    else:
        with console.status(f"Converting to ONNX IR..."):
            try:
                onnx_path = to_onnx(p)
                console.print(f"[green]✓[/green] Converted to ONNX IR")
            except ConversionError as e:
                console.print(f"[red]✗ Conversion failed:[/red] {e}")
                sys.exit(1)

    # ── Step 3: Analyze ONNX graph ───────────────────────────────────────────
    with console.status("Analyzing model graph..."):
        try:
            result = _analyze(onnx_path, original_format=fmt.value)
        except Exception as e:
            console.print(f"[red]✗ Analysis failed:[/red] {e}")
            sys.exit(1)

    # ── Operator summary ─────────────────────────────────────────────────────
    console.print()
    console.rule("[dim]Graph Summary[/dim]")

    op_str = "  ".join(
        f"[cyan]{op}[/cyan] ×{count}"
        for op, count in sorted(result.op_counts.items(), key=lambda x: -x[1])
    )
    console.print(f"[bold]Operators:[/bold] {result.total_ops} total")
    console.print(f"  {op_str}")
    console.print(f"[bold]Parameters:[/bold] {result.parameter_count:,}")

    # Input / output tensors
    console.print()
    for ti in result.input_tensors:
        shape_str = " × ".join(str(d) for d in ti.shape)
        console.print(
            f"[bold]Input:[/bold]  [cyan]{ti.name}[/cyan]  "
            f"[dim]{shape_str}  {ti.dtype}  {ti.bytes / 1024:.1f} KB[/dim]"
        )
    for ti in result.output_tensors:
        shape_str = " × ".join(str(d) for d in ti.shape)
        console.print(
            f"[bold]Output:[/bold] [cyan]{ti.name}[/cyan]  "
            f"[dim]{shape_str}  {ti.dtype}  {ti.bytes / 1024:.1f} KB[/dim]"
        )

    # Memory estimates
    console.print()
    console.rule("[dim]Memory Estimates[/dim]")
    console.print(
        f"[bold]Flash (weights, INT8):[/bold]  {result.flash_kb:.1f} KB\n"
        f"[bold]RAM   (peak activation):[/bold] {result.ram_kb:.1f} KB\n"
        f"[bold]Arena (TFLite Micro):[/bold]    {result.arena_kb:.1f} KB"
    )

    # ── Step 4: Compatibility report ─────────────────────────────────────────
    console.print()
    console.rule("[dim]Board Compatibility[/dim]")

    if mcu:
        try:
            target = load_target(mcu)
            compat_list = [check_compatibility(result.arena_kb, result.flash_kb, target)]
        except FileNotFoundError as e:
            console.print(f"[red]✗[/red] {e}")
            sys.exit(1)
    else:
        compat_list = check_all_targets(result.arena_kb, result.flash_kb)

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Board",  style="bold")
    table.add_column("Core",   style="dim")
    table.add_column("RAM",    justify="right")
    table.add_column("Flash",  justify="right")
    table.add_column("Arena",  justify="right")
    table.add_column("Status", justify="center")

    _status_icons = {
        FitStatus.FITS:      "[green]✓  FITS[/green]",
        FitStatus.TIGHT:     "[yellow]⚠  TIGHT[/yellow]",
        FitStatus.TOOSMALL:  "[red]✗  TOO SMALL[/red]",
    }

    for c in compat_list:
        t = c.target
        worst = (
            FitStatus.TOOSMALL if (
                c.ram_status == FitStatus.TOOSMALL or
                c.flash_status == FitStatus.TOOSMALL
            ) else (
                FitStatus.TIGHT if (
                    c.ram_status == FitStatus.TIGHT or
                    c.flash_status == FitStatus.TIGHT
                ) else FitStatus.FITS
            )
        )
        table.add_row(
            t.name,
            t.core,
            f"{t.ram_kb} KB",
            f"{t.flash_kb} KB",
            f"{c.arena_kb:.0f} KB",
            _status_icons[worst],
        )
        for w in c.warnings:
            table.add_row("", "", "", "", "", f"  [dim yellow]⚠ {w}[/dim yellow]")

    console.print(table)

    # ── JSON output ──────────────────────────────────────────────────────────
    if as_json:
        import json
        out = {
            "model": str(p),
            "format": fmt.value,
            "size_bytes": info["size_bytes"],
            "onnx_opset": result.onnx_opset,
            "op_counts": result.op_counts,
            "parameter_count": result.parameter_count,
            "flash_kb": result.flash_kb,
            "ram_kb": result.ram_kb,
            "arena_kb": result.arena_kb,
            "compatibility": [
                {
                    "target_id": c.target.id,
                    "fits": c.fits,
                    "ram_status": c.ram_status.value,
                    "flash_status": c.flash_status.value,
                    "warnings": c.warnings,
                }
                for c in compat_list
            ],
        }
        console.print_json(json.dumps(out, indent=2))


# ── edgeforge optimize ───────────────────────────────────────────────────────

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mcu", required=True, help="Target MCU profile ID.")
@click.option("--output", "-o", default=None, help="Output .onnx path.")
def optimize(model_path: str, mcu: str, output: str | None):
    """
    Optimize a model to fit the target MCU RAM/flash budget.
    Applies INT8 quantisation, operator fusion, and constant folding.
    """
    console.print(
        f"[bold]EdgeForge[/bold] optimizing [cyan]{model_path}[/cyan] "
        f"for [cyan]{mcu}[/cyan]..."
    )
    console.print("[dim]Phase 2 — not yet implemented.[/dim]")


# ── edgeforge compile ────────────────────────────────────────────────────────

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mcu", required=True, help="Target MCU profile ID.")
@click.option(
    "--rtos", default="none",
    type=click.Choice(["none", "freertos", "zephyr"]),
    help="RTOS for generated glue code.",
)
@click.option("--output-dir", "-o", default="edgeforge_output", help="Output directory.")
def compile(model_path: str, mcu: str, rtos: str, output_dir: str):
    """
    Compile a model to C/C++ files ready to drop into your firmware project.
    """
    console.print(
        f"[bold]EdgeForge[/bold] compiling [cyan]{model_path}[/cyan] "
        f"for [cyan]{mcu}[/cyan] (rtos=[cyan]{rtos}[/cyan])..."
    )
    console.print("[dim]Phase 3 — not yet implemented.[/dim]")


# ── edgeforge targets ────────────────────────────────────────────────────────

@main.command("targets")
@click.option("--mcu", default=None, help="Show details for a specific target.")
def list_targets(mcu: str | None):
    """List all supported MCU targets."""
    from .targets.loader import load_target, all_targets

    if mcu:
        try:
            t = load_target(mcu)
            _print_target_detail(t)
        except FileNotFoundError as e:
            console.print(f"[red]✗[/red] {e}")
            sys.exit(1)
        return

    table = Table(box=box.SIMPLE, header_style="bold dim")
    table.add_column("ID",      style="cyan bold")
    table.add_column("Name")
    table.add_column("Core",    style="dim")
    table.add_column("RAM",     justify="right")
    table.add_column("Flash",   justify="right")
    table.add_column("NPU",     justify="center")
    table.add_column("RTOS",    style="dim")

    for t in all_targets():
        rtos_parts = []
        if t.rtos_freertos: rtos_parts.append("FreeRTOS")
        if t.rtos_zephyr:   rtos_parts.append("Zephyr")
        table.add_row(
            t.id,
            t.name,
            t.core,
            f"{t.ram_kb} KB",
            f"{t.flash_kb} KB",
            "[green]✓[/green]" if t.npu else "[dim]—[/dim]",
            ", ".join(rtos_parts) or "—",
        )

    console.print()
    console.rule("[bold]EdgeForge Supported Targets[/bold]")
    console.print(table)


def _print_target_detail(t) -> None:
    console.print()
    console.rule(f"[bold]{t.name}[/bold]")
    console.print(f"  ID:      [cyan]{t.id}[/cyan]")
    console.print(f"  Vendor:  {t.vendor}")
    console.print(f"  Core:    {t.core}")
    console.print(f"  FPU:     {'yes' if t.fpu else 'no'}")
    console.print(f"  NPU:     {'yes' if t.npu else 'no'}")
    console.print(f"  RAM:     {t.ram_kb} KB")
    console.print(f"  Flash:   {t.flash_kb} KB")
    console.print(f"  Runtime: {t.runtime}")
    console.print(f"  CMSIS-NN: {'yes' if t.cmsis_nn else 'no'}")
    rtos = []
    if t.rtos_freertos: rtos.append("FreeRTOS")
    if t.rtos_zephyr:   rtos.append("Zephyr")
    console.print(f"  RTOS:    {', '.join(rtos) or '—'}")
    console.print(f"  Flags:   [dim]{t.compiler_flags}[/dim]")


if __name__ == "__main__":
    main()
