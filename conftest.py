"""Pytest configuration — patch targets dir to repo root."""
from pathlib import Path
from unittest.mock import patch
import pytest

TARGETS_DIR = Path(__file__).parent / "targets"

@pytest.fixture(autouse=True)
def patch_targets_dir(monkeypatch):
    from edgeforge.targets import loader
    monkeypatch.setattr(loader, "_targets_dir", lambda: TARGETS_DIR)
