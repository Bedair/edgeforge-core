"""
EdgeForge — Target Loader & Compatibility Reporter
Loads MCU target YAML profiles and evaluates model fit against hardware constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

import yaml


class FitStatus(str, Enum):
    FITS    = "fits"
    TIGHT   = "tight"       # fits but < 20% headroom
    TOOSMALL = "too_small"  # doesn't fit


@dataclass
class TargetProfile:
    id:              str
    name:            str
    vendor:          str
    core:            str
    fpu:             bool
    npu:             bool
    ram_kb:          int
    flash_kb:        int
    arena_default_kb: int
    cmsis_nn:        bool
    runtime:         str
    compiler_flags:  str
    rtos_freertos:   bool
    rtos_zephyr:     bool


@dataclass
class CompatResult:
    target:       TargetProfile
    ram_status:   FitStatus
    flash_status: FitStatus
    arena_kb:     float
    ram_used_pct: float
    flash_used_pct: float
    warnings:     list[str]

    @property
    def fits(self) -> bool:
        return (
            self.ram_status   != FitStatus.TOOSMALL and
            self.flash_status != FitStatus.TOOSMALL
        )


def _targets_dir() -> Path:
    """Locate the targets/ directory relative to this file."""
    # Works both in edgeforge-core layout and installed package
    candidates = [
        Path(__file__).parent.parent.parent / "targets",  # dev layout
        Path(__file__).parent.parent / "targets",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Cannot locate targets/ directory. "
        "Ensure you are running from the edgeforge-core repo root."
    )


def load_target(target_id: str) -> TargetProfile:
    """
    Load a specific MCU target profile by ID.

    Args:
        target_id: e.g. "stm32f407", "psoc6", "nrf52840"

    Returns:
        TargetProfile dataclass.

    Raises:
        FileNotFoundError: If no YAML file exists for the given ID.
    """
    path = _targets_dir() / f"{target_id}.yaml"
    if not path.exists():
        available = [p.stem for p in _targets_dir().glob("*.yaml")]
        raise FileNotFoundError(
            f"Unknown target '{target_id}'. "
            f"Available: {', '.join(sorted(available))}"
        )
    return _parse_yaml(path)


def all_targets() -> Iterator[TargetProfile]:
    """Iterate over all available MCU target profiles."""
    for p in sorted(_targets_dir().glob("*.yaml")):
        try:
            yield _parse_yaml(p)
        except Exception:
            continue  # skip malformed profiles silently


def check_compatibility(
    arena_kb: float,
    flash_kb: float,
    target: TargetProfile,
) -> CompatResult:
    """
    Check whether a model fits within a target MCU's memory budget.

    Args:
        arena_kb:  Required TFLite Micro arena size in KB.
        flash_kb:  Required flash (weights) in KB.
        target:    MCU target profile.

    Returns:
        CompatResult with fit status and usage percentages.
    """
    warnings: list[str] = []

    ram_used_pct   = arena_kb / target.ram_kb * 100
    flash_used_pct = flash_kb / target.flash_kb * 100

    # RAM fit
    if arena_kb > target.ram_kb:
        ram_status = FitStatus.TOOSMALL
        warnings.append(
            f"Arena ({arena_kb:.0f} KB) exceeds RAM ({target.ram_kb} KB). "
            "Apply quantisation or pruning."
        )
    elif ram_used_pct > 80:
        ram_status = FitStatus.TIGHT
        warnings.append(
            f"RAM usage is {ram_used_pct:.0f}% — very tight. "
            "Consider optimising before deployment."
        )
    else:
        ram_status = FitStatus.FITS

    # Flash fit
    if flash_kb > target.flash_kb * 0.6:  # models should use < 60% of flash
        flash_status = FitStatus.TOOSMALL
        warnings.append(
            f"Weights ({flash_kb:.0f} KB) exceed safe flash budget "
            f"({target.flash_kb * 0.6:.0f} KB). "
            "Quantise to INT8 to reduce size."
        )
    elif flash_used_pct > 40:
        flash_status = FitStatus.TIGHT
        warnings.append(
            f"Flash usage is {flash_used_pct:.0f}% — leaves little room for "
            "firmware code. Consider INT8 quantisation."
        )
    else:
        flash_status = FitStatus.FITS

    if not target.cmsis_nn:
        warnings.append(
            "CMSIS-NN not available — inference will be slow. "
            "Consider a Cortex-M4F or M7 target."
        )

    return CompatResult(
        target=target,
        ram_status=ram_status,
        flash_status=flash_status,
        arena_kb=arena_kb,
        ram_used_pct=ram_used_pct,
        flash_used_pct=flash_used_pct,
        warnings=warnings,
    )


def check_all_targets(arena_kb: float, flash_kb: float) -> list[CompatResult]:
    """Run compatibility check against all known targets."""
    return [
        check_compatibility(arena_kb, flash_kb, t)
        for t in all_targets()
    ]


# ── YAML parsing ─────────────────────────────────────────────────────────────

def _parse_yaml(path: Path) -> TargetProfile:
    with open(path) as f:
        d = yaml.safe_load(f)

    mem    = d.get("memory", {})
    inf    = d.get("inference", {})
    tc     = d.get("toolchain", {})
    rtos   = d.get("rtos", {})

    return TargetProfile(
        id=d["id"],
        name=d["name"],
        vendor=d["vendor"],
        core=d["core"],
        fpu=bool(d.get("fpu", False)),
        npu=bool(d.get("npu", False)),
        ram_kb=int(mem.get("ram_kb", 0)),
        flash_kb=int(mem.get("flash_kb", 0)),
        arena_default_kb=int(inf.get("arena_default_kb", 64)),
        cmsis_nn=bool(inf.get("cmsis_nn", False)),
        runtime=inf.get("runtime", "tflite-micro"),
        compiler_flags=tc.get("flags", ""),
        rtos_freertos=bool(rtos.get("freertos", False)),
        rtos_zephyr=bool(rtos.get("zephyr", False)),
    )
