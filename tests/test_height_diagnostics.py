"""height diagnostics: per-tile blob map + ray-plot renderers write PNGs."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")
pytest.importorskip("matplotlib")

from physhade.height.blob_separation import enforce_shadow_gap  # noqa: E402
from physhade.height.diagnostics import render_blob_map, render_ray_plot  # noqa: E402
from physhade.height.height_estimation import subpixel_flood_shadow_height_vectorized  # noqa: E402

AZ = 180.0  # rays march up the array


def _two_building_tile(h=80, w=80):
    building = np.zeros((h, w), np.uint8)
    shadow = np.zeros((h, w), np.float32)
    building[h - 3 :, :] = 1
    shadow[h - 23 : h - 3, 8:24] = 1.0
    shadow[h - 23 : h - 3, 44:60] = 1.0
    return shadow, building


def _diag_run():
    shadow, building = _two_building_tile()
    gap = enforce_shadow_gap(shadow, building, AZ)
    h_map, id_map, _, diag = subpixel_flood_shadow_height_vectorized(
        gap, building, AZ, 35.0, pixel_size=0.25, percentile=99, require_connection=False, diagnostics=True
    )
    return gap, building, h_map, id_map, diag


def test_render_ray_plot_writes_png(tmp_path):
    gap, building, _, _, diag = _diag_run()
    assert diag
    out = render_ray_plot(gap, building, diag, 0.25, AZ, tmp_path / "t_rays.png", percentile=99)
    assert out.exists() and out.stat().st_size > 1000


def test_render_blob_map_writes_png(tmp_path):
    import rasterio

    _, building, _, id_map, diag = _diag_run()
    h, w = id_map.shape
    rgb = np.random.default_rng(0).random((h, w, 3)).astype("float32")
    match_map = np.zeros_like(id_map, dtype="uint32")
    transform = rasterio.transform.from_origin(0, h * 0.25, 0.25, 0.25)
    trunc = {d["blob_id"] for d in diag if d["truncated"]}

    out = render_blob_map(rgb, id_map, match_map, None, transform, trunc, AZ, tmp_path / "t_map.png")
    assert out.exists() and out.stat().st_size > 1000
