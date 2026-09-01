"""Tests for the pseudo-shadow 'smear' physics prior.

Verifies the geometric core of the physics channel: shadow is cast AWAY from
the sun (opposite the solar azimuth), never onto the sunward side of a
footprint, and prior values stay in ``(0, 1]``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip(
    "geopandas", reason="smear module pulls the full geodata stack; covered by geo-enabled envs"
)
rasterio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

from physhade.physics import smear  # noqa: E402


def _write_footprint(path, size=64):
    """A single 8x8 building block near the centre of a 1 m/px grid."""
    mask = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    mask[c - 4 : c + 4, c - 4 : c + 4] = 1
    transform = Affine.translation(0, size) * Affine.scale(1, -1)
    meta = dict(driver="GTiff", height=size, width=size, count=1, dtype="uint8", transform=transform)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(mask, 1)
    return mask, c


def test_smear_module_importable():
    assert smear is not None


def test_shadow_cast_away_from_sun(tmp_path):
    """Sun in the east (azimuth 90) -> shadow falls to the west; the east
    side of the footprint stays clear and values are a decaying prior in (0, 1]."""
    size = 64
    fp_path = tmp_path / "footprint.tif"
    _, c = _write_footprint(fp_path, size)
    out_path = tmp_path / "smear.tif"

    smear.multi_shift_shadow_raster(
        raster_path=str(fp_path),
        azimuth_deg=90.0,  # sun due east
        length_mapunits=(2.0, 12.0),
        decay_type="delayed_gradient",
        out_tiff=str(out_path),
        out_shape=(size, size),
        mask_buildings=True,
    )

    with rasterio.open(out_path) as src:
        prior = src.read(1)

    assert prior.shape == (size, size)
    assert prior.max() <= 1.0 + 1e-6
    assert prior.min() >= 0.0
    assert prior.max() > 0.0  # something was written

    # column strictly east (right) of the footprint must be shadow-free
    east_strip = prior[c - 4 : c + 4, c + 6 :]
    assert east_strip.sum() == 0.0

    # west of the footprint must carry shadow
    west_strip = prior[c - 4 : c + 4, : c - 6]
    assert west_strip.sum() > 0.0
