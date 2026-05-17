"""Tests for MCU target loader."""
import pytest
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
TARGETS_DIR = REPO_ROOT / "targets"

def _patch_loader(monkeypatch):
    from edgeforge.targets import loader
    monkeypatch.setattr(loader, "_targets_dir", lambda: TARGETS_DIR)

def _require_targets():
    if not TARGETS_DIR.exists():
        pytest.skip(f"targets/ not found at {TARGETS_DIR}")

def test_all_targets_loads(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import all_targets
    targets = list(all_targets())
    ids = [t.id for t in targets]
    assert len(targets) >= 3
    assert "stm32f407" in ids and "psoc6" in ids and "nrf52840" in ids

def test_load_known_target(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target
    t = load_target("stm32f407")
    assert t.ram_kb == 192 and t.flash_kb == 1024 and t.cmsis_nn and not t.npu

def test_load_psoc6(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target
    t = load_target("psoc6")
    assert t.ram_kb == 288

def test_load_nrf52840(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target
    t = load_target("nrf52840")
    assert t.ram_kb == 256

def test_load_unknown_target(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target
    with pytest.raises(FileNotFoundError, match="Unknown target"):
        load_target("nonexistent_board_xyz")

def test_compatibility_fits(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target, check_compatibility, FitStatus
    t = load_target("stm32f407")
    r = check_compatibility(arena_kb=32, flash_kb=20, target=t)
    assert r.fits and r.ram_status == FitStatus.FITS

def test_compatibility_tight(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target, check_compatibility, FitStatus
    t = load_target("stm32f407")
    r = check_compatibility(arena_kb=170, flash_kb=50, target=t)
    assert r.ram_status == FitStatus.TIGHT

def test_compatibility_too_large(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import load_target, check_compatibility, FitStatus
    t = load_target("stm32f407")
    r = check_compatibility(arena_kb=500, flash_kb=2000, target=t)
    assert not r.fits and r.ram_status == FitStatus.TOOSMALL

def test_check_all_targets(monkeypatch):
    _require_targets(); _patch_loader(monkeypatch)
    from edgeforge.targets.loader import check_all_targets
    results = check_all_targets(arena_kb=32, flash_kb=20)
    assert len(results) >= 3
    assert all(r.fits for r in results)
