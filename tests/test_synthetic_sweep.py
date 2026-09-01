"""height_estimation --sweep: the azimuth x ray-length-percentile control (thesis Table 16)."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from physhade.height.height_estimation import run_sweep  # noqa: E402


def _footprint_tile(path, size=256):
    """One isolated 24x24 m building near the centre of a 0.25 m grid."""
    mask = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    mask[c - 48 : c + 48, c - 12 : c + 12] = 255  # tall-ish block
    transform = rasterio.transform.from_origin(0, size * 0.25, 0.25, 0.25)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)


def test_sweep_reproduces_p99_beats_p75(tmp_path):
    fp_dir = tmp_path / "footprints"
    fp_dir.mkdir()
    _footprint_tile(fp_dir / "block.tif")

    out = run_sweep(
        fp_dir,
        tmp_path / "sweep",
        azimuths=[120.0, 240.0],
        percentiles=[99.0, 75.0],
        true_length_m=10.0,
        solar_elevation=30.0,
    )

    summary = pd.read_csv(out / "azimuth_sweep_summary_metrics.csv").set_index("percentile")
    assert {99, 75} <= set(summary.index)
    assert np.isfinite(summary.loc[99, "rmse"])
    # the whole point of choosing p99: it is at least as accurate as p75
    assert summary.loc[99, "rmse"] <= summary.loc[75, "rmse"] + 1e-6
    assert summary.loc[99, "mean_abs_error"] < 1.0

    for name in ("azimuth_sweep_blob_metrics.csv", "azimuth_sweep_error_metrics.csv"):
        assert (out / name).exists()
    assert (out / "mean_error_vs_azimuth.png").exists()
