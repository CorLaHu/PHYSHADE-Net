"""Central path configuration and small run helpers for PHYSHADE-Net.

Every module imports its paths from here instead of hardcoding a
machine-specific location. Set the ``PHYSHADE_ROOT`` environment variable to
run the pipeline against a dataset that lives somewhere other than the repo
root; otherwise the repo root is found automatically.

The dataset is not distributed with this repository (see ``DATA.md``): it is
built from public Dutch geodata plus manual annotations. This module keeps
``torch`` out of its import path so the test suite can import it cheaply.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKERS = ("pyproject.toml", ".git")


def find_repo_root() -> Path:
    """Resolve the project root.

    ``PHYSHADE_ROOT`` wins if set. Otherwise walk up from this file looking
    for a ``pyproject.toml`` / ``.git`` marker. Falls back to the current
    working directory.
    """
    env = os.environ.get("PHYSHADE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if any((parent / m).exists() for m in _MARKERS):
            return parent
    return Path.cwd()


ROOT = find_repo_root()
DATA_DIR = ROOT / "Dataset"
OUTPUT_DIR = ROOT / "Output"
SOURCES_DIR = ROOT / "data_sources"


def find_latest_checkpoint(
    subdir: str,
    filename: str = "best_model.pth",
    pattern: str | None = None,
    root: Path | None = None,
) -> Path | None:
    """Newest checkpoint under ``OUTPUT_DIR/subdir`` by mtime, or ``None``.

    ``pattern`` overrides the default ``*/<filename>`` glob (e.g.
    ``"final_*/*/epoch*.pth"`` for the main-model runs).
    """
    base = (root or OUTPUT_DIR) / subdir
    if not base.exists():
        return None
    matches = sorted(base.glob(pattern or f"*/{filename}"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def resolve_checkpoint(
    cli_value: str | os.PathLike[str] | None,
    *,
    subdir: str,
    filename: str = "best_model.pth",
    pattern: str | None = None,
) -> Path:
    """Turn a ``--checkpoint`` CLI value into a concrete, existing path.

    Explicit value: used as-is (error if missing). Otherwise the most recent
    matching checkpoint is auto-discovered. If nothing is found, exit with an
    actionable message rather than a late ``FileNotFoundError``.
    """
    if cli_value:
        path = Path(cli_value).expanduser()
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")
        return path
    found = find_latest_checkpoint(subdir, filename, pattern)
    if found is None:
        raise SystemExit(
            f"no checkpoint under {OUTPUT_DIR / subdir} - run the upstream "
            f"training stage first, or pass an explicit --checkpoint / "
            f"--base-checkpoint path"
        )
    return found
