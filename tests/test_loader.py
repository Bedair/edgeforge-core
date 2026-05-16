"""Tests for MCU target loader."""
import pytest
from pathlib import Path

# Repo root is two levels up from this file: tests/ -> repo root
REPO_ROOT   = Path(__file__).parent.parent
TARGETS_DIR = REPO_ROOT / "targets"


def _patch_loader(monkeypatch):
    """Point the loader _targets_dir() at the real targets/ folder."""
    from edgeforge.targets import loader
    monkeypatch.setattr(loader, "_targets_dir", lambda: TARGETS_DIR)


def _require_targets():
    if not TARGETS_DIR.exists():
        pytest.skip(f"targets/ not found at {TARGETS_DIR}")


def test_all_targets_loads(monkeypatch):
    """all_targets() must yield at least the 3 built-in profiles."""
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import all_targets
    targets = list(all_targets())
    ids = [t.id for t in targets]
    assert len(targets) >= 3
    assert "stm32f407" in ids
    assert "psoc6"     in ids
    assert "nrf52840"  in ids


def test_load_known_target(monkeypatch):
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target
    t = load_target("stm32f407")
    assert t.id       == "stm32f407"
    assert t.ram_kb   == 192
    assert t.flash_kb == 1024
    assert t.cmsis_nn is True
    assert t.fpu      is True
    assert t.npu      is False
    assert t.rtos_freertos is True
    assert t.rtos_zephyr   is True


def test_load_psoc6(monkeypatch):
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target
    t = load_target("psoc6")
    assert t.ram_kb == 288
    assert t.flash_kb == 2048


def test_load_nrf52840(monkeypatch):
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target
    t = load_target("nrf52840")
    assert t.ram_kb == 256
    assert t.flash_kb == 1024


def test_load_unknown_target(monkeypatch):
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target
    with pytest.raises(FileNotFoundError, match="Unknown target"):
        load_target("nonexistent_board_xyz")


def test_compatibility_fits(monkeypatch):
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target, check_compatibility, FitStatus
    t = load_target("stm32f407")
    result = check_compatibility(arena_kb=32, flash_kb=20, target=t)
    assert result.fits         is True
    assert result.ram_status   == FitStatus.FITS
    assert result.flash_status == FitStatus.FITS
    assert result.ram_used_pct < 20


def test_compatibility_tight(monkeypatch):
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target, check_compatibility, FitStatus
    t = load_target("stm32f407")
    result = check_compatibility(arena_kb=170, flash_kb=50, target=t)
    assert result.ram_status == FitStatus.TIGHT
    assert len(result.warnings) > 0


def test_compatibility_too_large(monkeypatch):
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target, check_compatibility, FitStatus
    t = load_target("stm32f407")
    result = check_compatibility(arena_kb=500, flash_kb=2000, target=t)
    assert result.fits       is False
    assert result.ram_status == FitStatus.TOOSMALL


def test_check_all_targets(monkeypatch):
    _require_targets()
    _patch_loader(monkeypatch)
    from edgeforge.targets.loader import check_all_targets
    results = check_all_targets(arena_kb=32, flash_kb=20)
    assert len(results) >= 3
    ids = [r.target.id for r in results]
    assert "stm32f407" in ids
    assert "psoc6"     in ids
    assert "nrf52840"  in ids
    for r in results:
        assert r.fits is True, f"{r.target.id} should fit a 32KB model"
