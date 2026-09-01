"""Thesis feature-parity checks for the height pipeline (assign-and-break, p99,
sun-edge rays + truncation, is_highest_blob / blob-weighted aggregation)."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip(
    "geopandas", reason="height modules pull the full geodata stack; covered by geo-enabled envs"
)
pytest.importorskip("pvlib")
pytest.importorskip("skimage")
pytest.importorskip("sklearn")

from physhade.height.blob_separation import (  # noqa: E402
    _enforce_pixel_gap_reference,
    enforce_pixel_gap,
    enforce_shadow_gap,
)
from physhade.height.height_estimation import (  # noqa: E402
    subpixel_flood_shadow_height_vectorized,
)

# azimuth 180 -> (az+180)%360 = 0 -> rays march along -y (up the array),
# sun-facing edge pixels are the ones whose neighbour below is a building.
AZ_UP = 180.0
ELEV_45 = 45.0  # tan(45) = 1 -> est_height == shadow_length_m


def test_enforce_pixel_gap_matches_reference():
    rng = np.random.default_rng(0)
    raster = rng.integers(0, 4, size=(40, 40)).astype(np.uint16)
    np.testing.assert_array_equal(enforce_pixel_gap(raster), _enforce_pixel_gap_reference(raster))


def test_enforce_shadow_gap_separates_two_buildings():
    h = w = 80
    building = np.zeros((h, w), np.uint8)
    shadow = np.zeros((h, w), np.float32)

    # two buildings side by side, each with a shadow column directly above it
    building[50:55, 10:25] = 1
    building[50:55, 45:60] = 1
    shadow[20:50, 10:25] = 1
    shadow[20:50, 45:60] = 1

    gap = enforce_shadow_gap(shadow, building, AZ_UP)
    labels = sorted(int(x) for x in np.unique(gap) if x)
    assert len(labels) == 2, f"expected 2 blobs, got {labels}"

    # the two shadow columns must not be 4-connected through a shared label
    from scipy.ndimage import label as nd_label

    _, n_components = nd_label(gap > 0)
    assert n_components == 2


def _stack_two_blobs(h=80, w=80):
    """A ragged (triangular) shadow that touches the left border + a clean
    interior rectangle, both sitting on a building strip along the bottom."""
    building = np.zeros((h, w), np.uint8)
    shadow = np.zeros((h, w), np.float32)
    building[h - 3 :, :] = 1

    # blob A: right triangle, column x in [0, 20) has shadow (x+1) px tall, touches col 0
    for x in range(0, 20):
        shadow[h - 4 - x : h - 3, x] = 1.0

    # blob B: clean 20 px tall rectangle, interior columns, not touching any border
    shadow[h - 23 : h - 3, 40:60] = 1.0
    return shadow, building


def test_vectorized_returns_4_tuple_and_flags_truncation():
    shadow, building = _stack_two_blobs()
    out = subpixel_flood_shadow_height_vectorized(
        shadow,
        building,
        AZ_UP,
        ELEV_45,
        pixel_size=0.25,
        percentile=99,
        require_connection=False,
        diagnostics=True,
    )
    assert len(out) == 4
    _, _, _, diag = out
    trunc = {d["blob_id"]: d["truncated"] for d in diag}
    assert any(trunc.values()) and not all(trunc.values()), trunc


def test_p99_recovers_known_length_and_exceeds_p75():
    shadow, building = _stack_two_blobs()

    def est(pct):
        hm, idm, _, diag = subpixel_flood_shadow_height_vectorized(
            shadow,
            building,
            AZ_UP,
            ELEV_45,
            pixel_size=0.25,
            percentile=pct,
            require_connection=False,
            diagnostics=True,
        )
        # the clean interior rectangle is the non-truncated blob
        d = [x for x in diag if not x["truncated"]]
        assert d, "no non-truncated blob"
        return d[0]["est_height_m"]

    # rectangle is 20 px tall -> 20 * 0.25 m * tan(45) = 5.0 m
    assert est(99) == pytest.approx(5.0, abs=0.3)
    # on the ragged triangle blob p99 must exceed p75
    tri99 = next(x["est_height_m"] for x in _diag(shadow, building, 99) if x["truncated"])
    tri75 = next(x["est_height_m"] for x in _diag(shadow, building, 75) if x["truncated"])
    assert tri99 > tri75


def _diag(shadow, building, pct):
    return subpixel_flood_shadow_height_vectorized(
        shadow,
        building,
        AZ_UP,
        ELEV_45,
        pixel_size=0.25,
        percentile=pct,
        require_connection=False,
        diagnostics=True,
    )[3]


def test_is_highest_blob_and_weighted_average(tmp_path):
    from physhade.height.height_pipeline import _aggregate_and_write

    # building X: two blobs (est 8, 10) -> the est=10 one is "highest"; building Y: one blob
    records = [
        dict(
            image="T1",
            exp_id="M",
            fold="0",
            ckpt="c",
            blob_id=1,
            azimuth=1.0,
            solar_elevation=30.0,
            n_pixels=50,
            percentile_ray_length_px=1.0,
            ray_len=1.0,
            shadow_length=1.0,
            est_height=8.0,
            true_height=10.0,
            delta_height=-2.0,
            matched_building_id="X",
        ),
        dict(
            image="T1",
            exp_id="M",
            fold="0",
            ckpt="c",
            blob_id=2,
            azimuth=1.0,
            solar_elevation=30.0,
            n_pixels=50,
            percentile_ray_length_px=1.0,
            ray_len=1.0,
            shadow_length=1.0,
            est_height=10.0,
            true_height=10.0,
            delta_height=0.0,
            matched_building_id="X",
        ),
        dict(
            image="T1",
            exp_id="M",
            fold="0",
            ckpt="c",
            blob_id=3,
            azimuth=1.0,
            solar_elevation=30.0,
            n_pixels=50,
            percentile_ray_length_px=1.0,
            ray_len=1.0,
            shadow_length=1.0,
            est_height=12.0,
            true_height=9.0,
            delta_height=3.0,
            matched_building_id="Y",
        ),
    ]
    df = _aggregate_and_write(records, [], tmp_path)
    highest = df[df["is_highest_blob"]]
    assert set(zip(highest["matched_building_id"], highest["blob_id"])) == {("X", 2), ("Y", 3)}

    per_blob = pd.read_csv(tmp_path / "per_blob_metrics.csv")
    assert "Is Highest Blob?" in per_blob.columns

    model = pd.read_csv(tmp_path / "weighted_model_metrics.csv")
    assert (model["exp_id"] == "AVERAGE_WEIGHTED").any()
    # accepted rows: (X, blob2, delta 0) and (Y, blob3, delta 3) -> MAE = 1.5
    avg = model.loc[model["exp_id"] == "AVERAGE_WEIGHTED", "mean_absolute_error"].iloc[0]
    assert avg == pytest.approx(1.5, abs=1e-6)
