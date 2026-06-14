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
@click.option("--mcu", required=True, help="Target MCU profile ID (e.g. nrf52840, psoc6, stm32f407).")
@click.option("--rtos", default="none",
              type=click.Choice(["none", "freertos", "zephyr"]),
              help="RTOS for generated glue code (default: none).")
@click.option(
    "--target", "build_target", default="cmake",
    type=click.Choice(["cmake", "arduino"]),
    help=(
        "Output format.\n"
        "cmake   = C/C++ source files for any CMake/Makefile project (default).\n"
        "arduino = model_data.h byte array for Arduino IDE + starter .ino sketch."
    )
)
@click.option("--output-dir", "-o", default="edgeforge_output",
              help="Output directory for generated files (cmake) or sketch folder (arduino).")
def compile(model_path: str, mcu: str, rtos: str,
            build_target: str, output_dir: str):
    """Compile a model to firmware-ready code for your target.

    \b
    For a CMake/Makefile firmware project:
      edgeforge compile models/gesture_model_opt.onnx --mcu=nrf52840 --rtos=freertos

    \b
    For Arduino IDE:
      edgeforge compile models/gesture_model_opt.onnx --mcu=nrf52840 --target=arduino
      edgeforge compile models/gesture_model_opt.onnx --mcu=nrf52840 --target=arduino -o MySketch/
    """
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
    console.print(f"Mode:   {build_target}")
    console.print(f"Output: {output_dir}")

    # ── Arduino target ────────────────────────────────────────────────────────
    if build_target == "arduino":
        _compile_arduino(p, target, output_dir)
        return

    # ── CMake target (default) ────────────────────────────────────────────────
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
        console.print(
            f"[NOTE] Arena fits in CCM SRAM -- set EDGEFORGE_USE_CCM=1 for better performance"
        )
    console.print()
    console.print(f"{OK} Output written to: {result.output_dir}")


def _onnx_to_tflite(p: Path) -> bytes | None:
    """
    Convert an ONNX model to TFLite flatbuffer bytes using onnx2tf.
    Returns the float32 .tflite bytes, or None on failure.
    """
    import os, tempfile

    try:
        import onnx2tf
    except ImportError:
        console.print(
            f"\n{FAIL} onnx2tf not installed.\n"
            f"  Fix: pip install onnx2tf\n"
            f"  Then retry the compile command."
        )
        return None

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    try:
        with tempfile.TemporaryDirectory(prefix="edgeforge_tflite_") as tmp:
            onnx2tf.convert(
                input_onnx_file_path=str(p),
                output_folder_path=tmp,
                non_verbose=True,
            )
            tmp_path = Path(tmp)

            # Prefer float32 over float16
            float32 = [f for f in tmp_path.glob("*.tflite") if "float16" not in f.name]
            any_tflite = list(tmp_path.glob("*.tflite"))

            candidates = float32 or any_tflite
            if not candidates:
                console.print(f"\n{FAIL} onnx2tf ran but produced no .tflite file")
                return None

            data = candidates[0].read_bytes()

            # Verify the TFLite is valid by checking the magic bytes
            if len(data) < 8:
                console.print(f"\n{FAIL} Generated TFLite file is too small ({len(data)} bytes)")
                return None

            console.print(f"\n{OK} Converted to TFLite ({len(data):,} bytes)")
            return data

    except Exception as e:
        console.print(f"\n{FAIL} onnx2tf conversion failed: {e}")
        return None


# TFLite builtin op code -> (name, resolver method)
_TFLITE_OPS: dict[int, tuple[str, str]] = {
    0:   ("ADD",                  "AddAdd"),
    1:   ("AVERAGE_POOL_2D",      "AddAveragePool2D"),
    2:   ("CONCATENATION",        "AddConcatenation"),
    3:   ("CONV_2D",              "AddConv2D"),
    4:   ("DEPTHWISE_CONV_2D",    "AddDepthwiseConv2D"),
    6:   ("DEQUANTIZE",           "AddDequantize"),
    9:   ("FULLY_CONNECTED",      "AddFullyConnected"),
    14:  ("LOGISTIC",             "AddLogistic"),
    17:  ("MAX_POOL_2D",          "AddMaxPool2D"),
    18:  ("MUL",                  "AddMul"),
    19:  ("RELU",                 "AddRelu"),
    21:  ("RELU6",                "AddRelu6"),
    22:  ("RESHAPE",              "AddReshape"),
    25:  ("SOFTMAX",              "AddSoftmax"),
    27:  ("SVDF",                 "AddSvdf"),
    28:  ("TANH",                 "AddTanh"),
    34:  ("PAD",                  "AddPad"),
    36:  ("GATHER",               "AddGather"),
    39:  ("TRANSPOSE",            "AddTranspose"),
    40:  ("MEAN",                 "AddMean"),
    41:  ("SUB",                  "AddSub"),
    42:  ("DIV",                  "AddDiv"),
    43:  ("SQUEEZE",              "AddSqueeze"),
    45:  ("STRIDED_SLICE",        "AddStridedSlice"),
    47:  ("EXP",                  "AddExp"),
    49:  ("SPLIT",                "AddSplit"),
    53:  ("CAST",                 "AddCast"),
    60:  ("PADV2",                "AddPadV2"),
    65:  ("SLICE",                "AddSlice"),
    69:  ("TILE",                 "AddTile"),
    70:  ("EXPAND_DIMS",          "AddExpandDims"),
    74:  ("SUM",                  "AddSum"),
    75:  ("SQRT",                 "AddSqrt"),
    83:  ("PACK",                 "AddPack"),
    88:  ("UNPACK",               "AddUnpack"),
    114: ("QUANTIZE",             "AddQuantize"),
    126: ("BATCH_MATMUL",         "AddBatchMatMul"),
}


def _parse_tflite_ops(tflite_bytes: bytes) -> list[tuple[int, str, str]]:
    """
    Parse a TFLite flatbuffer and return the exact list of ops used.
    Returns list of (builtin_code, op_name, resolver_method).

    Uses onnx2tf's schema_generated.py if available (most accurate),
    falls back to flatbuffers-based parsing.
    """
    import tempfile, sys

    # Try onnx2tf schema parser first (most reliable)
    try:
        import onnx2tf, os, importlib.util

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

        with tempfile.TemporaryDirectory(prefix="edgeforge_schema_") as tmp:
            # Run a dummy convert to get schema_generated.py
            # OR find it from onnx2tf package
            import onnx2tf as o2t
            schema_path = Path(o2t.__file__).parent / "utils" / "schema_generated.py"
            if not schema_path.exists():
                # Try to generate it by converting a dummy model
                raise FileNotFoundError("schema_generated.py not found in onnx2tf")

            spec = importlib.util.spec_from_file_location("schema_generated", schema_path)
            schema = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(schema)

            buf = bytearray(tflite_bytes)
            model = schema.ModelT.InitFromPackedBuf(buf, 0)

            seen = set()
            ops_used = []
            for subgraph in model.subgraphs:
                for op in subgraph.operators:
                    oc = model.operatorCodes[op.opcodeIndex]
                    code = oc.builtinCode if oc.builtinCode > 127 else oc.deprecatedBuiltinCode
                    if code not in seen:
                        seen.add(code)
                        name, method = _TFLITE_OPS.get(code, (f"UNKNOWN_{code}", None))
                        ops_used.append((code, name, method))

            return ops_used

    except Exception:
        pass

    # Fallback: use flatbuffers to parse the operator_codes vector
    try:
        import struct

        root_off = struct.unpack_from('<I', tflite_bytes, 0)[0]
        # Walk vtable to find operator_codes (field index 2 -> vtable slot 4)
        vt_off = root_off - struct.unpack_from('<i', tflite_bytes, root_off)[0]
        vt_size = struct.unpack_from('<H', tflite_bytes, vt_off)[0]

        ops_used = []
        if vt_size >= 8:
            f2 = struct.unpack_from('<H', tflite_bytes, vt_off + 4)[0]
            if f2 > 0:
                vec_off = root_off + f2
                vec_ref = struct.unpack_from('<i', tflite_bytes, vec_off)[0]
                vec_abs = vec_off + vec_ref
                count   = struct.unpack_from('<I', tflite_bytes, vec_abs)[0]

                if 0 < count < 200:  # sanity check
                    seen = set()
                    for i in range(count):
                        item_off = vec_abs + 4 + i * 4
                        item_ref = struct.unpack_from('<i', tflite_bytes, item_off)[0]
                        item_abs = item_off + item_ref
                        oc_vt_off  = item_abs - struct.unpack_from('<i', tflite_bytes, item_abs)[0]
                        oc_vt_size = struct.unpack_from('<H', tflite_bytes, oc_vt_off)[0]

                        code = None
                        # Field 1 = deprecated_builtin_code (int8)
                        if oc_vt_size >= 6:
                            f1 = struct.unpack_from('<H', tflite_bytes, oc_vt_off + 4)[0]
                            if f1 > 0:
                                code = tflite_bytes[item_abs + f1]
                        # Field 4 = builtin_code (int32, overrides if > 127)
                        if oc_vt_size >= 12:
                            f4 = struct.unpack_from('<H', tflite_bytes, oc_vt_off + 10)[0]
                            if f4 > 0:
                                val = struct.unpack_from('<i', tflite_bytes, item_abs + f4)[0]
                                if val > 127:
                                    code = val

                        if code is not None and code not in seen:
                            seen.add(code)
                            name, method = _TFLITE_OPS.get(code, (f"UNKNOWN_{code}", None))
                            ops_used.append((code, name, method))

        return ops_used

    except Exception:
        return []


def _compile_arduino(p: Path, target, output_dir: str) -> None:
    """
    Generate Arduino IDE output:
      - model_data.h  : TFLite flatbuffer as C byte array
      - <name>.ino    : Arduino sketch with correct resolver for this specific model
    """
    import shutil

    sketch_name = p.stem
    out = Path(output_dir) / sketch_name
    out.mkdir(parents=True, exist_ok=True)

    console.print()

    # ── Step 1: Get TFLite bytes ──────────────────────────────────────────────
    if p.suffix.lower() == ".tflite":
        tflite_bytes = p.read_bytes()
        console.print(f"{OK} TFLite model read ({len(tflite_bytes):,} bytes)")
    else:
        with console.status("Converting ONNX -> TFLite..."):
            tflite_bytes = _onnx_to_tflite(p)
        if tflite_bytes is None:
            sys.exit(1)

    # ── Step 2: Parse exact ops from the TFLite flatbuffer ───────────────────
    with console.status("Parsing TFLite ops..."):
        ops = _parse_tflite_ops(tflite_bytes)

    if ops:
        known   = [(c, n, m) for c, n, m in ops if m is not None]
        unknown = [(c, n, m) for c, n, m in ops if m is None]
        console.print(f"{OK} Detected {len(ops)} ops: {', '.join(n for _, n, _ in ops)}")
        if unknown:
            console.print(f"{WARN} Unknown ops (no resolver method): {[n for _,n,_ in unknown]}")
    else:
        console.print(f"{WARN} Could not parse ops -- using safe default set")
        # Safe default covering most float32 DS-CNN style models
        known = [
            (3,  "CONV_2D",           "AddConv2D"),
            (4,  "DEPTHWISE_CONV_2D", "AddDepthwiseConv2D"),
            (9,  "FULLY_CONNECTED",   "AddFullyConnected"),
            (19, "RELU",              "AddRelu"),
            (22, "RESHAPE",           "AddReshape"),
            (25, "SOFTMAX",           "AddSoftmax"),
            (34, "PAD",               "AddPad"),
            (40, "MEAN",              "AddMean"),
        ]

    # ── Step 3: Write model_data.h ────────────────────────────────────────────
    op_names_comment = ", ".join(n for _, n, _ in known)
    header_lines = [
        "// EdgeForge Generated File -- DO NOT EDIT",
        f"// Model:   {p.stem}",
        f"// Target:  {target.name}",
        f"// Size:    {len(tflite_bytes):,} bytes",
        f"// Ops:     {op_names_comment}",
        f"// Command: edgeforge compile {p.name} --mcu={target.id} --target=arduino",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"const unsigned int model_data_len = {len(tflite_bytes)}U;",
        "",
        "alignas(8) const uint8_t model_data[] = {",
    ]
    cols = 12
    for i in range(0, len(tflite_bytes), cols):
        chunk   = tflite_bytes[i:i+cols]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        comma   = "," if i + cols < len(tflite_bytes) else ""
        header_lines.append(f"    {hex_str}{comma}")
    header_lines += ["};", ""]
    (out / "model_data.h").write_text("\n".join(header_lines), encoding="utf-8")
    console.print(f"{OK} model_data.h written")

    # ── Step 4: Generate .ino with correct resolver block ─────────────────────
    n_ops        = len(known)
    resolver_decl = f"static tflite::MicroMutableOpResolver<{n_ops}> resolver;"
    resolver_adds = "\n    ".join(f"resolver.{m}();" for _, _, m in known)
    arena_kb      = max(target.ram_kb // 8, 32)  # 1/8 of RAM, min 32 KB

    # Check if the model uses IMU input (shape [1,1,50,3] style)
    uses_imu = "gesture" in p.stem.lower() or "imu" in p.stem.lower()

    if uses_imu:
        sensor_includes = "#include <Arduino_LSM9DS1.h>\n"
        sensor_init = """\
    if (!IMU.begin()) {
        Serial.println("[FAIL] LSM9DS1 init failed");
        while (1);
    }
    Serial.print("[OK]  LSM9DS1 ready (");
    Serial.print(IMU.accelerationSampleRate());
    Serial.println(" Hz)");"""
        sensor_loop = """\
    float ax, ay, az;
    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax, ay, az);
        g_window[g_window_idx][0] = ax;
        g_window[g_window_idx][1] = ay;
        g_window[g_window_idx][2] = az;
        g_window_idx = (g_window_idx + 1) % WINDOW_SAMPLES;
        if (g_window_idx == 0) g_window_full = true;
    }
    if (g_window_full && g_window_idx == 0) {
        run_inference();
    }"""
        sensor_globals = """\
#define SAMPLE_RATE_HZ   50
#define WINDOW_SAMPLES   50
#define N_AXES           3
static float         g_window[WINDOW_SAMPLES][N_AXES];
static int           g_window_idx  = 0;
static bool          g_window_full = false;
static unsigned long g_last_sample_ms = 0;"""
        infer_fill = """\
    float *inp = input_tensor->data.f;
    for (int i = 0; i < WINDOW_SAMPLES; i++) {
        int idx = (g_window_idx + i) % WINDOW_SAMPLES;
        inp[i * N_AXES + 0] = g_window[idx][0];
        inp[i * N_AXES + 1] = g_window[idx][1];
        inp[i * N_AXES + 2] = g_window[idx][2];
    }"""
        loop_body = f"""\
    unsigned long now = millis();
    if ((now - g_last_sample_ms) < (1000 / SAMPLE_RATE_HZ)) return;
    g_last_sample_ms = now;
    {sensor_loop}"""
    else:
        sensor_includes = ""
        sensor_init = "    // Fill input_tensor->data.f with your sensor data before calling run_inference()"
        sensor_globals = ""
        infer_fill = "    // TODO: fill input_tensor->data.f with your input data"
        loop_body = "    run_inference();  // TODO: replace with your sensor read + inference trigger"

    ino_content = f"""\
/*
 * EdgeForge Generated Sketch
 * Model:  {p.stem}
 * Target: {target.name}
 * Ops:    {op_names_comment}
 *
 * Libraries:
 *   1. git clone https://github.com/tensorflow/tflite-micro-arduino-examples
 *         into: C:\\Users\\YOUR_NAME\\Documents\\Arduino\\libraries\\Arduino_TensorFlowLite
 *   2. Tools -> Manage Libraries -> Arduino_LSM9DS1 -> Install  (if using IMU)
 *
 * Serial: 115200 baud
 */

{sensor_includes}#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model_data.h"

#define N_CLASSES        3
#define ARENA_SIZE       ({arena_kb} * 1024)
{sensor_globals}

const char *CLASS_NAMES[] = {{ "idle", "shake", "tap" }};

// Resolver with exactly the ops this model uses -- auto-generated by EdgeForge
{resolver_decl}
static uint8_t              tensor_arena[ARENA_SIZE];
static const tflite::Model *tfl_model    = nullptr;
static tflite::MicroInterpreter *interpreter  = nullptr;
static TfLiteTensor         *input_tensor  = nullptr;
static TfLiteTensor         *output_tensor = nullptr;

void setup() {{
    Serial.begin(115200);
    while (!Serial && millis() < 3000);
    Serial.println("EdgeForge -- {p.stem}");

{sensor_init}

    {resolver_adds}

    tfl_model = tflite::GetModel(model_data);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {{
        Serial.println("[FAIL] Schema version mismatch");
        while (1);
    }}
    interpreter = new tflite::MicroInterpreter(
        tfl_model, resolver, tensor_arena, ARENA_SIZE
    );
    if (interpreter->AllocateTensors() != kTfLiteOk) {{
        Serial.println("[FAIL] AllocateTensors failed");
        while (1);
    }}
    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
    Serial.print("[OK] Model loaded  arena_used=");
    Serial.println(interpreter->arena_used_bytes());
    Serial.println("idle=still  shake=fast move  tap=knock");
}}

void loop() {{
    {loop_body}
}}

void run_inference() {{
{infer_fill}

    unsigned long t0 = micros();
    if (interpreter->Invoke() != kTfLiteOk) {{
        Serial.println("[WARN] Invoke failed");
        return;
    }}
    unsigned long ms = (micros() - t0) / 1000;

    float *out = output_tensor->data.f;
    int best = 0;
    for (int i = 1; i < N_CLASSES; i++)
        if (out[i] > out[best]) best = i;

    Serial.print("[INFER] "); Serial.print(CLASS_NAMES[best]);
    Serial.print("  (");
    for (int i = 0; i < N_CLASSES; i++) {{
        Serial.print(CLASS_NAMES[i]); Serial.print("="); Serial.print(out[i], 2);
        if (i < N_CLASSES - 1) Serial.print("  ");
    }}
    Serial.print(")  "); Serial.print(ms); Serial.println(" ms");
}}
"""
    (out / f"{sketch_name}.ino").write_text(ino_content, encoding="utf-8")
    console.print(f"{OK} {sketch_name}.ino written (resolver: {n_ops} ops auto-detected)")

    # ── Report ────────────────────────────────────────────────────────────────
    console.rule("Generated Files")
    console.print(f"  {OK} model_data.h       ({len(tflite_bytes):,} bytes)")
    console.print(f"  {OK} {sketch_name}.ino  (resolver auto-generated)")
    console.print()
    console.rule("Next steps (Arduino IDE)")
    console.print(f"  1. File -> Open -> {out}/{sketch_name}.ino")
    console.print(f"  2. Tools -> Board -> Arduino Nano 33 BLE Sense")
    console.print(f"  3. Upload -> Serial Monitor at 115200 baud")
    console.print()
    console.print(f"{OK} Output: {out}")



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


# ── edgeforge benchmark ──────────────────────────────────────────────────────

@main.command()
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--mcu", required=True, help="Target MCU profile ID.")
@click.option("--compiler", default="arm-none-eabi-gcc",
              help="Cross-compiler to use for compile check.")
def benchmark(output_dir: str, mcu: str, compiler: str):
    """Validate generated files compile for the target MCU and report size estimates."""
    import subprocess
    import shutil
    from edgeforge.targets.loader import load_target

    out = Path(output_dir)
    console.rule("EdgeForge Benchmark")

    try:
        target = load_target(mcu)
    except FileNotFoundError as e:
        console.print(f"{FAIL} {e}"); sys.exit(1)

    console.print(f"\nOutput dir: {out}")
    console.print(f"Target:     {target.name}")
    console.print(f"Compiler:   {compiler}\n")

    # ── Check required files exist ────────────────────────────────────────────
    required = ["model.h", "model.c", "memory_config.h",
                "inference_runner.h", "inference_runner.c"]
    console.rule("File check")
    all_present = True
    for fname in required:
        fpath = out / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            console.print(f"  {OK} {fname:<30} ({size_kb:.1f} KB)")
        else:
            console.print(f"  {FAIL} {fname} -- MISSING")
            all_present = False

    if not all_present:
        console.print(f"\n{FAIL} Missing files -- re-run edgeforge compile")
        sys.exit(1)

    # ── Compiler check ────────────────────────────────────────────────────────
    console.print()
    console.rule("Compile check")

    gcc_available = shutil.which(compiler) is not None

    if not gcc_available:
        console.print(
            f"  {WARN} {compiler} not found -- skipping compile check.\n"
            f"  Install from: https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain\n"
            f"  On Windows: winget install Arm.GnuArmEmbeddedToolchain"
        )
    else:
        # Compile model.c and inference_runner.c (syntax + type check only, no link)
        flags = target.compiler_flags.split() + [
            "-mthumb", "-std=c11", "-Os", "-Wall", "-Wextra",
            "-DEDGEFORGE_COMPILE_CHECK",
            f"-I{out}",
            "-c",
        ]

        files_to_check = [
            str(out / "model.c"),
            str(out / "inference_runner.c"),
        ]
        if (out / "rtos_glue.c").exists():
            files_to_check.append(str(out / "rtos_glue.c"))

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            all_ok = True
            for src in files_to_check:
                fname = Path(src).name
                obj   = str(Path(tmp) / (fname + ".o"))
                cmd   = [compiler] + flags + [src, "-o", obj]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    console.print(f"  {OK} {fname}")
                else:
                    console.print(f"  {FAIL} {fname}")
                    for line in result.stderr.strip().splitlines():
                        console.print(f"      {line}")
                    all_ok = False

            if all_ok:
                console.print(f"\n  {OK} All files compiled without errors")
            else:
                console.print(f"\n  {FAIL} Compile errors found -- check output above")
                sys.exit(1)

    # ── Static analysis -- read memory config defines ─────────────────────────
    console.print()
    console.rule("Memory report")

    mem_config = (out / "memory_config.h").read_text(encoding="utf-8")

    def extract_define(content: str, name: str) -> str:
        import re
        m = re.search(rf"#define\s+{name}\s+(\S+)", content)
        return m.group(1).rstrip("U") if m else "?"

    arena_bytes = extract_define(mem_config, "EDGEFORGE_ARENA_SIZE")
    alignment   = extract_define(mem_config, "EDGEFORGE_ARENA_ALIGNMENT")
    ram_kb      = extract_define(mem_config, "EDGEFORGE_TARGET_RAM_KB")
    flash_kb    = extract_define(mem_config, "EDGEFORGE_TARGET_FLASH_KB")
    cmsis_nn    = extract_define(mem_config, "EDGEFORGE_CMSIS_NN")

    model_h = (out / "model.h").read_text(encoding="utf-8")
    weight_bytes = extract_define(model_h, "EDGEFORGE_WEIGHT_BYTES")
    input_size   = extract_define(model_h, "EDGEFORGE_INPUT_SIZE")
    output_size  = extract_define(model_h, "EDGEFORGE_OUTPUT_SIZE")
    is_quant     = extract_define(model_h, "EDGEFORGE_IS_QUANTIZED")

    try:
        arena_kb_f  = int(arena_bytes) / 1024
        weight_kb_f = int(weight_bytes) / 1024
        ram_used    = int(arena_bytes) / (int(ram_kb) * 1024) * 100
        flash_used  = int(weight_bytes) / (int(flash_kb) * 1024) * 100
    except (ValueError, ZeroDivisionError):
        arena_kb_f = weight_kb_f = ram_used = flash_used = 0.0

    console.print(f"  Arena size:     {arena_bytes} bytes ({arena_kb_f:.1f} KB)")
    console.print(f"  Arena align:    {alignment} bytes")
    console.print(f"  Weight storage: {weight_bytes} bytes ({weight_kb_f:.1f} KB)")
    console.print(f"  Input buffer:   {input_size} bytes")
    console.print(f"  Output buffer:  {output_size} bytes")
    console.print(f"  Quantised:      {'yes' if is_quant == '1' else 'no'}")
    console.print(f"  CMSIS-NN:       {'yes' if cmsis_nn == '1' else 'no'}")
    console.print()
    console.print(f"  RAM  used: {ram_used:.1f}% of {ram_kb} KB")
    console.print(f"  Flash used: {flash_used:.1f}% of {flash_kb} KB")
    console.print()

    if ram_used < 80 and flash_used < 60:
        console.print(f"{OK} Model fits {target.id} with comfortable headroom")
    elif ram_used < 100 and flash_used < 100:
        console.print(f"{WARN} Model fits {target.id} but headroom is tight")
    else:
        console.print(f"{FAIL} Model does not fit {target.id} -- re-run edgeforge optimize")
        sys.exit(1)

    # ── Next steps ────────────────────────────────────────────────────────────
    console.print()
    console.rule("Next steps")
    console.print(f"  1. Copy edgeforge_output/ into your {target.name} firmware project")
    console.print(f"  2. Add TFLite Micro as a dependency")
    console.print(f"  3. Include validation/main.c for a ready-made test harness")
    console.print(f"  4. Flash and check UART output")

