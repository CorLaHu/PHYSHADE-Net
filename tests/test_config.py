"""Tests for physhade.config helpers and module import-safety."""

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physhade import config  # noqa: E402


def test_find_repo_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PHYSHADE_ROOT", str(tmp_path))
    assert config.find_repo_root() == tmp_path.resolve()


def test_find_repo_root_marker_walk(monkeypatch):
    monkeypatch.delenv("PHYSHADE_ROOT", raising=False)
    root = config.find_repo_root()
    assert (root / "pyproject.toml").exists()


def test_find_latest_checkpoint_missing(tmp_path):
    assert config.find_latest_checkpoint("base_model", root=tmp_path) is None


def test_find_latest_checkpoint_picks_newest(tmp_path):
    old = tmp_path / "base_model" / "2020_01_01_00_00"
    new = tmp_path / "base_model" / "2026_01_01_00_00"
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "best_model.pth").write_bytes(b"x")
    import os
    import time

    os.utime(old / "best_model.pth", (time.time() - 1000, time.time() - 1000))
    assert config.find_latest_checkpoint("base_model", root=tmp_path) == new / "best_model.pth"


def test_resolve_checkpoint_empty_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    with pytest.raises(SystemExit):
        config.resolve_checkpoint(None, subdir="base_model")


_ALL_MODULES = [
    m.name for m in pkgutil.walk_packages(importlib.import_module("physhade").__path__, prefix="physhade.")
]


@pytest.mark.parametrize("modname", _ALL_MODULES)
def test_modules_import_without_side_effects(modname):
    """Importing any physhade submodule must not raise (no work at import time)."""
    try:
        importlib.import_module(modname)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing for {modname}: {exc}")
