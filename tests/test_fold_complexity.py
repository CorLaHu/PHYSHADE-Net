"""fold_complexity: per-tile edge / shadow-coverage / contrast, per-fold summary."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("cv2")

from physhade.inference import fold_complexity as fc  # noqa: E402


def test_tile_metrics_ranges():
    rng = np.random.default_rng(0)
    rgb = rng.random((64, 64, 3)).astype("float32")
    mask = np.zeros((64, 64), bool)
    mask[:32] = True  # half shadow
    ed, sc, ct = fc._tile_metrics(rgb, mask)
    assert 0 <= ed <= 1
    assert sc == pytest.approx(0.5)
    assert ct >= 0


def _tile(path, fill, mask_val):
    img = np.full((32, 32, 3), fill, dtype="uint8")
    tr = rasterio.transform.from_origin(0, 32, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff", height=32, width=32, count=3, dtype="uint8", transform=tr
    ) as d:
        d.write(img.transpose(2, 0, 1))
    m = np.zeros((32, 32), "uint8")
    m[:mask_val] = 255
    with rasterio.open(
        str(path).replace("image", "annotated_shadow"),
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=1,
        dtype="uint8",
        transform=tr,
    ) as d:
        d.write(m, 1)


def test_run_writes_per_fold_summary(tmp_path, monkeypatch):
    data = tmp_path / "train"
    (data / "image").mkdir(parents=True)
    (data / "annotated_shadow").mkdir(parents=True)
    _tile(data / "image" / "a.tif", 40, 8)
    _tile(data / "image" / "b.tif", 200, 20)

    monkeypatch.setattr(fc, "_folds", lambda *a, **k: [["a"], ["b"]])
    out = fc.run(data, tmp_path / "fc", k=2, seed=42)

    per_image = pd.read_csv(out / "image_complexity_metrics.csv")
    assert set(fc._METRICS) <= set(per_image.columns)
    assert sorted(per_image["Fold"].unique().tolist()) == [1, 2]
    assert (out / "fold_complexity_summary.csv").exists()
