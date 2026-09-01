"""training_preprocessing --val: regenerate val_set/smear_shadow + pairs/singles."""

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("skimage")

from physhade.data import training_preprocessing as tp  # noqa: E402


def _footprint(path, size=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size, size), "uint8")
    mask[40:88, 50:70] = 255
    tr = rasterio.transform.from_origin(155000, 463000 + size * 0.25, 0.25, 0.25)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs="EPSG:28992",
        transform=tr,
    ) as d:
        d.write(mask, 1)


def test_preprocess_val_writes_smear_and_pairs(tmp_path, monkeypatch):
    root = tmp_path / "val_set"
    for stem in ("Summer_Tile6", "Winter_Tile6", "Summer_Tile9"):
        _footprint(root / "building_footprint" / f"{stem}.tif")
        _footprint(root / "georef_image" / f"{stem}.tif")

    monkeypatch.setattr(
        "physhade.height.height_pipeline.solar_angle_lookup",
        lambda *a, **k: {
            "Summer_Tile6": (135.0, 40.0),
            "Winter_Tile6": (150.0, 15.0),
            "Summer_Tile9": (120.0, 45.0),
        },
    )
    tp.preprocess_val(root, tmp_path / "s.shp", tmp_path / "w.shp")

    smears = sorted(p.name for p in (root / "smear_shadow").glob("*.tif"))
    assert smears == ["Summer_Tile6.tif", "Summer_Tile9.tif", "Winter_Tile6.tif"]
    with rasterio.open(root / "smear_shadow" / "Summer_Tile6.tif") as s:
        vals = s.read(1)
    assert vals.max() <= 1.0 + 1e-6 and (vals > 0).any()

    pairs = (root / "pairs.txt").read_text().split()
    singles = (root / "singles.txt").read_text().split()
    assert pairs == ["Winter_Tile6", "Summer_Tile6"]
    assert singles == ["Summer_Tile9"]
