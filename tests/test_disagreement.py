"""disagreement: A-only / B-only / agreement overlays + per-uid fraction stats."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("matplotlib")

from physhade.inference.disagreement import _masks_for, _parse_pairs, run  # noqa: E402


def test_parse_pairs():
    assert _parse_pairs("A:B, C:D") == [("A", "B"), ("C", "D")]
    with pytest.raises(SystemExit):
        _parse_pairs("bogus")


def _write(path, arr, bands=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    tr = rasterio.transform.from_origin(0, arr.shape[-1], 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[-2],
        width=arr.shape[-1],
        count=bands,
        dtype="float32",
        transform=tr,
    ) as d:
        d.write(arr.astype("float32") if bands > 1 else arr.astype("float32")[None])


def test_run_writes_overlay_and_stats(tmp_path):
    h = w = 32
    a = np.zeros((h, w))
    a[:16] = 1  # top half
    b = np.zeros((h, w))
    b[8:24] = 1  # middle band -> overlaps rows 8-15
    masks = tmp_path / "inferred_masks"
    _write(masks / "MODEL_A" / "fold_0" / "ck" / "t1.tif", a)
    _write(masks / "MODEL_B" / "fold_0" / "ck" / "t1.tif", b)
    img_dir = tmp_path / "img"
    _write(img_dir / "t1.tif", np.full((3, h, w), 0.5), bands=3)

    assert set(_masks_for(masks, "MODEL_A")) == {"t1"}
    written = run(masks, img_dir, [("MODEL_A", "MODEL_B")], tmp_path / "out")

    assert len(written) == 1
    stats = pd.read_csv(written[0])
    row = stats.iloc[0]
    assert row["agreement_pixels"] == 8 * 32  # a=rows0-15, b=rows8-23 -> overlap rows 8-15
    assert 0 <= row["disagreement_fraction"] <= 1
    assert (written[0].parent / "t1_disagreement.png").exists()
