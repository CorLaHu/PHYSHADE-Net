"""ablation_tables: reads runs.csv, writes the segmentation-ablation stat CSVs."""

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from physhade.inference.ablation_tables import compute_ci95, run  # noqa: E402


def _runs_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    configs = {
        "BASE_BCE": 0.55,
        "BASE_DICE": 0.60,
        "RGB_BCE50_DICE50": 0.62,
        "RGBS_BCE50_DICE50": 0.85,
        "HYB_DICE_PHYS10": 0.86,
        "PHYS_ATT_0.5": 0.63,
    }
    for exp_id, base in configs.items():
        for fold in range(1, 6):
            rows.append(
                {
                    "exp_id": exp_id,
                    "fold": fold,
                    "dice": base + rng.normal(0, 0.01),
                    "precision": base + rng.normal(0, 0.01),
                    "recall": base + rng.normal(0, 0.01),
                    "loss": 1 - base + rng.normal(0, 0.01),
                    "ran_epoch": int(rng.integers(40, 120)),
                }
            )
        rows.append(
            {
                "exp_id": exp_id,
                "fold": "avg",
                "dice": base,
                "precision": base,
                "recall": base,
                "loss": 1 - base,
                "ran_epoch": 80,
            }
        )
    return pd.DataFrame(rows)


def test_compute_ci95_widens_with_variance():
    lo_tight, hi_tight = compute_ci95(pd.Series([0.5, 0.5, 0.5, 0.5]))
    lo_wide, hi_wide = compute_ci95(pd.Series([0.1, 0.9, 0.2, 0.8]))
    assert (hi_wide - lo_wide) > (hi_tight - lo_tight)


def test_run_writes_ranked_summary_and_paired_tests(tmp_path):
    csv = tmp_path / "runs.csv"
    _runs_frame().to_csv(csv, index=False)
    written = run(csv, tmp_path / "tables")

    assert "01_fold_statistics.csv" in written
    assert "07_final_summary.csv" in written

    final = pd.read_csv(tmp_path / "tables" / "07_final_summary.csv")
    assert final["Mean Dice"].is_monotonic_decreasing
    assert final.iloc[0]["Experiment ID"] in {"HYB_DICE_PHYS10", "RGBS_BCE50_DICE50"}

    paired = pd.read_csv(tmp_path / "tables" / "04_subset_b_paired_ablation.csv")
    assert "p-value" in paired.columns
    assert np.isfinite(paired["p-value"]).all()
    # RGBS beats RGB by a wide margin -> significant positive delta
    row = paired.loc[paired["Comparison"] == "BCE50_DICE50"].iloc[0]
    assert row["Delta Mean"] > 0.1 and row["Significant?"] == "Yes"


def test_partial_runs_csv_still_produces_core_tables(tmp_path):
    frame = _runs_frame()
    frame = frame[frame["exp_id"].isin(["HYB_DICE_PHYS10"])]  # only one config
    csv = tmp_path / "runs.csv"
    frame.to_csv(csv, index=False)
    written = run(csv, tmp_path / "t")
    assert "01_fold_statistics.csv" in written
    assert "07_final_summary.csv" in written
    assert "04_subset_b_paired_ablation.csv" not in written
