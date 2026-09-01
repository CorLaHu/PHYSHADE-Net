"""build_heights: actual_height = b3_h_70p - 70th-pct(DTM over footprint bbox +5 m)."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")

from shapely.geometry import Point, box  # noqa: E402

from physhade.config import SOURCES_DIR  # noqa: E402
from physhade.data.build_heights import _extend_bbox, _sample_dtm, run  # noqa: E402

_DTM_DIR = Path(r"C:\Users\Lars\Desktop\cum")
_REF_GPKG = SOURCES_DIR / "heights_validation_area.gpkg"


def test_extend_bbox_grows_by_buffer():
    b = _extend_bbox(Point(100, 200).buffer(1.0), buffer_m=5.0).bounds
    assert b[0] == pytest.approx(94.0) and b[2] == pytest.approx(106.0)


def test_sample_dtm_takes_percentile_over_box(tmp_path):
    ramp = np.tile(np.linspace(0, 100, 100, dtype="float32"), (100, 1))  # 0..100 west->east
    path = tmp_path / "dtm.tif"
    transform = rasterio.transform.from_origin(0, 100, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(ramp, 1)
    with rasterio.open(path) as src:
        val = _sample_dtm(box(0, 0, 100, 100), [src], pct=70.0)
    assert val == pytest.approx(np.percentile(ramp, 70), abs=1.0)


@pytest.mark.skipif(
    not _DTM_DIR.exists() or not _REF_GPKG.exists(), reason="reference DTM tiles / heights gpkg not present"
)
def test_rederive_matches_committed_validation_gpkg(tmp_path):
    ref = gpd.read_file(_REF_GPKG)[["identificatie", "actual_height"]].dropna()
    out = run(_REF_GPKG, _DTM_DIR, tmp_path / "h.gpkg")  # use the gpkg itself as the BAG source
    got = gpd.read_file(out)[["identificatie", "actual_height"]].dropna()

    m = ref.merge(got, on="identificatie", suffixes=("_ref", "_got"))
    assert len(m) >= 10
    diff = (m["actual_height_ref"] - m["actual_height_got"]).abs()
    assert diff.median() < 0.3
    assert (diff < 1.0).mean() > 0.8
