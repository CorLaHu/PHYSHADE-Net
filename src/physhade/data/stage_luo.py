"""Stage the Luo et al. cross-domain comparison tiles for qualitative inference.

    python -m physhade.data.stage_luo

Extracts ``luo_validation.tar`` into
``Dataset/main_model/luo/{img,masks_machine}`` (the layout ``quick_inference`` expects;
``lineup/`` is created by the inference stage).
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from physhade.config import DATA_DIR, SOURCES_DIR


def stage(tar_path: Path, out_dir: Path) -> dict[str, int]:
    tmp = out_dir.parent / "_luo_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(tmp)  # noqa: S202 - trusted local archive

    ds_dir = next(p for p in tmp.iterdir() if p.is_dir() and p.name.startswith("dataset "))
    counts = {}
    for src_sub, dst_sub in [("img", "img"), ("masks_machine", "masks_machine")]:
        src = ds_dir / src_sub
        if not src.is_dir():
            continue
        dst = out_dir / dst_sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        counts[dst_sub] = len(list(dst.iterdir()))
    shutil.rmtree(tmp, ignore_errors=True)
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tar", type=Path, default=SOURCES_DIR / "luo_validation.tar")
    p.add_argument("--out", type=Path, default=DATA_DIR / "main_model" / "luo")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.tar.exists():
        raise SystemExit(f"not found: {args.tar}")
    print(f"luo staged: {stage(args.tar, args.out)}")


if __name__ == "__main__":
    main()
