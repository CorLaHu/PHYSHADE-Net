"""Rebuild the PHYSHADE-Net ``Dataset/`` tree from the source archives.

    python -m physhade.data.build_dataset            # uses data_sources/  (see docs/pipeline.md)
    python -m physhade.data.build_dataset --staging /some/dir

Produces, under ``Dataset/``:

    main_model/all/{raw,image,class_masks,building_footprint,annotated_shadow,
                    annotation_export,human/accepted}
    main_model/shpfiles/          (the two PDOK flight-overview shapefiles, unzipped)
    val_set/{annotated_shadows,smear_shadow,ann}
    aisd/{Train412,Val51,Test51}/{shadow,mask}

``training_preprocessing`` then adds ``all/smear_shadow`` and the ``train/`` split;
``build_base_dataset`` turns ``aisd/`` into ``base_model/``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import tarfile
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from physhade.config import DATA_DIR, SOURCES_DIR
from physhade.data.split_annotations import split_imagery

BUILDING_CLASSES = ("Buildings", "Building")
SHADOW_CLASSES = ("Shadows from buildings", "Shadow from Building")


# --------------------------------------------------------------------------- #
# Supervisely bitmap decoding (fallback when masks_machine/ is absent)
# --------------------------------------------------------------------------- #
def _decode_bitmap(bitmap: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(bitmap["data"]))
    png = Image.open(io.BytesIO(raw)).convert("RGBA")
    return np.array(png)[..., 3] > 0


def _paint_class(ann: dict, class_titles: tuple[str, ...]) -> np.ndarray:
    h, w = ann["size"]["height"], ann["size"]["width"]
    canvas = np.zeros((h, w), bool)
    for obj in ann["objects"]:
        if obj.get("classTitle") not in class_titles:
            continue
        m = _decode_bitmap(obj["bitmap"])
        c0, r0 = obj["bitmap"]["origin"]
        canvas[r0 : r0 + m.shape[0], c0 : c0 + m.shape[1]] |= m
    return canvas


def _extract_tar(tar_path: Path, out_dir: Path) -> Path | None:
    """Extract a Supervisely project tar once; return its ``ann/`` dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if not marker.exists():
        with tarfile.open(tar_path) as tf:
            tf.extractall(out_dir)  # noqa: S202 - trusted local archives
        marker.touch()
    for p in out_dir.rglob("ann"):
        if p.is_dir() and any(f.suffix == ".json" for f in p.iterdir()):
            return p
    return None


# --------------------------------------------------------------------------- #
# build stages
# --------------------------------------------------------------------------- #
def _raw_and_image(staging: Path, all_dir: Path) -> int:
    raw_dir, image_dir = all_dir / "raw", all_dir / "image"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(staging / "PHYSHADE_Image_Pairs.zip") as zf:
        for name in zf.namelist():
            if not name.endswith(".tif.tiff"):
                continue
            data = zf.read(name)
            base = Path(name).name
            (raw_dir / base).write_bytes(data)
            (image_dir / base).write_bytes(data)
            n += 1
    return n


def _class_masks(staging: Path, all_dir: Path, annotation_tars: list[Path]) -> tuple[int, int]:
    for sub in ("class_masks", "building_footprint", "annotated_shadow", "annotation_export"):
        shutil.rmtree(all_dir / sub, ignore_errors=True)
        (all_dir / sub).mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n = n_shadow = 0
    for tar in annotation_tars:
        ann_dir = _extract_tar(tar, staging / f"_tar_{tar.stem}")
        if ann_dir is None:
            raise SystemExit(f"no ann/ dir inside {tar.name}")
        mm_dir = ann_dir.parent / "masks_machine"
        for js in sorted(ann_dir.glob("*.json")):
            stem = js.name.replace(".tif.tiff.json", "")
            if stem in seen:
                continue
            seen.add(stem)
            mm_png = mm_dir / f"{stem}.tif.png"
            if mm_png.exists():
                cls = np.array(Image.open(mm_png)).astype(np.uint8)
                if cls.ndim == 3:
                    cls = cls[..., 0]
            else:
                ann = json.loads(js.read_text())
                buildings = _paint_class(ann, BUILDING_CLASSES)
                shadows = _paint_class(ann, SHADOW_CLASSES)
                cls = np.zeros(buildings.shape, np.uint8)
                cls[buildings] = 1
                cls[shadows] = 2
            Image.fromarray(cls).save(all_dir / "class_masks" / f"{stem}.png")
            n += 1
            if (cls == 2).any():
                n_shadow += 1

        # keep the export verbatim for provenance
        exp_root = ann_dir.parent
        for sub in exp_root.iterdir():
            if not sub.is_dir():
                continue
            dst = all_dir / "annotation_export" / tar.stem / sub.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in sub.iterdir():
                if f.suffix in (".png", ".json") or f.name.endswith(".tif.tiff"):
                    shutil.copy2(f, dst / f.name)
    return n, n_shadow


def _accepted(all_dir: Path, accepted_file: Path | None) -> int:
    """Populate ``all/human/accepted`` with the shadow GeoTIFF of every accepted tile.

    Accepted = listed in ``accepted_file`` if given, else every tile whose class
    mask carries shadow pixels (the thesis 'N of 52' filter).
    """
    acc_dir = all_dir / "human" / "accepted"
    shutil.rmtree(acc_dir, ignore_errors=True)
    acc_dir.mkdir(parents=True, exist_ok=True)
    shadow_dir = all_dir / "annotated_shadow"

    if accepted_file and accepted_file.exists():
        stems = [s.strip() for s in accepted_file.read_text().splitlines() if s.strip()]
    else:
        stems = []
        for png in sorted((all_dir / "class_masks").glob("*.png")):
            if (np.array(Image.open(png)) == 2).any():
                stems.append(png.stem)

    n = 0
    for stem in stems:
        tif = shadow_dir / f"{stem}.tif"
        if tif.exists():
            shutil.copy2(tif, acc_dir / f"{stem}.tif")
            n += 1
    return n


def _shapefiles(staging: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for zname in ("LR_BMO_2023.zip", "BMO2023.zip"):
        with zipfile.ZipFile(staging / zname) as zf:
            zf.extractall(out_dir)


def _aisd(staging: Path, out_root: Path) -> dict[str, int]:
    src_root = staging / "AISD"
    counts = {}
    for split in ("Train412", "Val51", "Test51"):
        for sub in ("shadow", "mask"):
            src, dst = src_root / split / sub, out_root / split / sub
            if not src.is_dir():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                shutil.copy2(f, dst / f.name)
            counts[f"{split}/{sub}"] = len(list(dst.iterdir()))
    return counts


def _val_set(staging: Path, val_dir: Path) -> dict[str, int]:
    """Extract the val-set inputs from ``val_source.zip``; keep any existing
    ``georef_image`` / ``summary`` / ``class_masks_val`` in place."""
    src_zip = staging / "val_source.zip"
    if not src_zip.exists():
        print(f"  val_source.zip not found in {staging} - skipping val_set")
        return {}
    tmp = staging / "_val_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(src_zip) as zf:
        zf.extractall(tmp)
    base = tmp / "last_val_set" / "import"
    mapping = {
        "annotated_shadows": "annotated_shadows",
        "smear_shadow/accepted": "smear_shadow",
        "building_footprint/accepted": "building_footprint",
        "ann": "ann",
    }
    counts = {}
    for src_sub, dst_sub in mapping.items():
        src = base / src_sub
        if not src.is_dir():
            continue
        dst = val_dir / dst_sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.suffix == ".tif":
                shutil.copy2(f, dst / f.name)
        counts[dst_sub] = len(list(dst.iterdir()))
    shutil.rmtree(tmp, ignore_errors=True)
    return counts


# --------------------------------------------------------------------------- #
def build(staging: Path, out: Path, accepted_file: Path | None) -> None:
    all_dir = out / "main_model" / "all"

    tars = sorted(staging.glob("proper_annotations_*.tar"))
    if not tars:
        raise SystemExit(
            f"no proper_annotations_*.tar in {staging} - see docs/pipeline.md for the "
            f"expected data_sources/ layout"
        )

    n_raw = _raw_and_image(staging, all_dir)
    print(f"raw + image tiles: {n_raw}")

    n_cls, n_shadow = _class_masks(staging, all_dir, tars)
    print(f"class masks: {n_cls} ({n_shadow} with shadow labels) from {len(tars)} exports")

    n_split = split_imagery(all_dir)
    print(f"split into building/shadow GeoTIFFs: {n_split}")

    n_acc = _accepted(all_dir, accepted_file)
    print(f"accepted tiles: {n_acc}")

    _shapefiles(staging, out / "main_model" / "shpfiles")
    print("shapefiles unzipped -> main_model/shpfiles/")

    aisd_counts = _aisd(staging, out / "aisd")
    print(f"AISD copied: {aisd_counts}")

    val_counts = _val_set(staging, out / "val_set")
    print(f"val_set: {val_counts}")

    # regenerate the val smear priors from the (shipped) footprint rasters so
    # smear_shadow/ isn't trusted from the zip
    val_dir = out / "val_set"
    if (val_dir / "building_footprint").is_dir():
        try:
            from physhade.data.training_preprocessing import preprocess_val

            shp = out / "main_model" / "shpfiles"
            preprocess_val(
                val_dir,
                shp / "2023_beeldmiddenoverzicht_lrl.shp",
                shp / "2023_beeldmiddenoverzicht_hrl.shp",
            )
        except Exception as exc:  # noqa: BLE001 - keep the build going; smear from the zip is the fallback
            print(f"  val smear rebuild skipped ({exc}); using the zip's smear_shadow/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--staging", type=Path, default=SOURCES_DIR, help="source-archive dir (default: data_sources/)"
    )
    p.add_argument("--out", type=Path, default=DATA_DIR, help="dataset root (default: Dataset/)")
    p.add_argument(
        "--accepted-file",
        type=Path,
        default=Path(__file__).with_name("accepted_tiles.txt"),
        help="one stem per line; if absent, accepted = tiles with shadow labels",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.staging.is_dir():
        raise SystemExit(f"staging dir not found: {args.staging}")
    build(args.staging, args.out, args.accepted_file)


if __name__ == "__main__":
    main()
