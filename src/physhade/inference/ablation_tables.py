"""Stage B5c: the segmentation ablation statistics tables (thesis Tables 4-15).

    python -m physhade.inference.ablation_tables --runs-csv Output/main_model/runs.csv

Pure pandas over ``runs.csv`` - no inference. Writes eight CSVs to
``Output/main_model/ablation_tables/``: per-fold stats, per-subset summaries
(baseline / supervised RGB-vs-RGBS / physics-loss / hybrid) with paired
``ttest_rel`` comparisons and 95% CIs, a final ranked summary, and an
epochs-to-converge ablation. Subsets whose experiments are absent from
``runs.csv`` are skipped with a note, so a partial run still produces the
per-fold and final tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from physhade.config import OUTPUT_DIR

_METRIC_AGG = {
    "Mean Dice": ("dice", "mean"),
    "Std. Dice": ("dice", "std"),
    "Mean Precision": ("precision", "mean"),
    "Std. Precision": ("precision", "std"),
    "Mean Recall": ("recall", "mean"),
    "Std. Recall": ("recall", "std"),
    "Mean Loss": ("loss", "mean"),
    "Std. Loss": ("loss", "std"),
}

_RGB_IDS = ["RGB_BCE30_DICE70", "RGB_BCE50_DICE50", "RGB_BCE70_DICE30"]
_RGBS_IDS = ["RGBS_BCE30_DICE70", "RGBS_BCE50_DICE50", "RGBS_BCE70_DICE30"]
_PHYS_BASELINE = {"BCE": "BASE_BCE", "DICE": "BASE_DICE", "ATT": "RGB_BCE50_DICE50"}
_HYB_BASELINE = "RGBS_BCE50_DICE50"


def compute_ci95(series: pd.Series) -> tuple[float, float]:
    """95% CI of the mean: ``mean +/- 1.96 * std / sqrt(n)``."""
    mean, std, n = series.mean(), series.std(), len(series)
    ci = 1.96 * std / np.sqrt(n) if n > 1 else np.nan
    return mean - ci, mean + ci


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("exp_id").agg(**_METRIC_AGG, **{"Mean Epochs Run": ("ran_epoch", "mean")})
    agg = agg.reset_index().rename(columns={"exp_id": "Experiment ID"})
    lo, hi = zip(*df.groupby("exp_id")["dice"].apply(compute_ci95), strict=False)
    agg["Dice CI95 Lower"], agg["Dice CI95 Upper"] = lo, hi
    return agg


def _paired(a: np.ndarray, b: np.ndarray) -> dict | None:
    """Paired ``b - a`` delta stats + t-test p-value, or ``None`` if unpairable."""
    if len(a) != len(b) or len(a) < 2:
        return None
    d = np.asarray(b, float) - np.asarray(a, float)
    lo, hi = compute_ci95(pd.Series(d))
    return {
        "Delta Mean": float(d.mean()),
        "Delta Std.": float(d.std()),
        "Delta CI95 Lower": lo,
        "Delta CI95 Upper": hi,
        "p-value": float(ttest_rel(b, a).pvalue),
        "Significant?": "Yes" if ttest_rel(b, a).pvalue < 0.05 else "No",
    }


def _vals(fold_df: pd.DataFrame, exp_id: str, col: str) -> np.ndarray:
    return fold_df.loc[fold_df["exp_id"] == exp_id, col].to_numpy()


def run(runs_csv: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(runs_csv)
    fold_df = df[df["fold"].astype(str) != "avg"].copy()
    fold_df["fold"] = pd.to_numeric(fold_df["fold"], errors="coerce")
    present = set(fold_df["exp_id"].unique())
    written: list[str] = []

    def _write(name: str, frame: pd.DataFrame) -> None:
        frame.round(4).to_csv(out_dir / name, index=False, float_format="%.4f")
        written.append(name)

    # 1. per-fold statistics
    fold_stats = fold_df.groupby("fold").agg(**_METRIC_AGG).reset_index().rename(columns={"fold": "Fold"})
    _write("01_fold_statistics.csv", fold_stats)

    # 2. subset A - unsupervised-loss baselines
    if {"BASE_BCE", "BASE_DICE"} <= present:
        _write(
            "02_subset_a_baseline.csv", _summary(fold_df[fold_df["exp_id"].isin(["BASE_BCE", "BASE_DICE"])])
        )

    # 3 + 4. subset B - RGB vs RGBS (prior channel)
    b_ids = [e for e in _RGB_IDS + _RGBS_IDS if e in present]
    if b_ids:
        _write("03_subset_b_supervised.csv", _summary(fold_df[fold_df["exp_id"].isin(b_ids)]))
    paired_b = []
    for rgb, rgbs in zip(_RGB_IDS, _RGBS_IDS, strict=True):
        res = _paired(_vals(fold_df, rgb, "dice"), _vals(fold_df, rgbs, "dice"))
        if res is None:
            continue
        row = {"Comparison": rgb.replace("RGB_", "")} | res
        for m in ("precision", "recall"):
            dm = _vals(fold_df, rgbs, m) - _vals(fold_df, rgb, m)
            row[f"Delta Mean {m.capitalize()}"] = float(dm.mean())
        paired_b.append(row)
    if paired_b:
        _write("04_subset_b_paired_ablation.csv", pd.DataFrame(paired_b))

    # 5. subset C - physics-guided loss vs its baseline
    c_rows = []
    for phys in sorted(e for e in present if e.startswith("PHYS_")):
        key = next((k for k in _PHYS_BASELINE if k in phys), None)
        base = _PHYS_BASELINE.get(key)
        res = _paired(_vals(fold_df, base, "dice"), _vals(fold_df, phys, "dice")) if base in present else None
        if res is None:
            continue
        c_rows.append(
            {"Physics Config": phys.replace("PHYS_", ""), "Baseline": base}
            | {"Physics Mean Dice": _vals(fold_df, phys, "dice").mean()}
            | res
        )
    if c_rows:
        _write("05_subset_c_physics_ablation.csv", pd.DataFrame(c_rows))

    # 6. subset D - hybrid vs RGBS_BCE50_DICE50
    d_rows = []
    if _HYB_BASELINE in present:
        for hyb in sorted(e for e in present if e.startswith("HYB_")):
            res = _paired(_vals(fold_df, _HYB_BASELINE, "dice"), _vals(fold_df, hyb, "dice"))
            if res is None:
                continue
            d_rows.append(
                {"Hybrid Config": hyb.replace("HYB_", "").replace("_", " ")}
                | {"Hybrid Mean Dice": _vals(fold_df, hyb, "dice").mean()}
                | res
            )
    if d_rows:
        _write("06_subset_d_hybrid_ablation.csv", pd.DataFrame(d_rows))

    # 7. final summary, ranked by Dice
    final = _summary(fold_df).sort_values("Mean Dice", ascending=False)
    _write("07_final_summary.csv", final)

    # 8. epochs-to-converge ablation (reuses the pairings above)
    epoch_rows = []
    for label, a_id, b_id in [
        *[("B", rgb, rgbs) for rgb, rgbs in zip(_RGB_IDS, _RGBS_IDS, strict=True)],
        *[
            ("C", _PHYS_BASELINE.get(next((k for k in _PHYS_BASELINE if k in e), ""), ""), e)
            for e in sorted(present)
            if e.startswith("PHYS_")
        ],
        *[("D", _HYB_BASELINE, e) for e in sorted(present) if e.startswith("HYB_")],
    ]:
        if a_id not in present or b_id not in present:
            continue
        res = _paired(_vals(fold_df, a_id, "ran_epoch"), _vals(fold_df, b_id, "ran_epoch"))
        if res is None:
            continue
        epoch_rows.append({"Subset": label, "Comparison": f"{a_id} vs {b_id}"} | res)
    if epoch_rows:
        _write("08_epoch_ablation.csv", pd.DataFrame(epoch_rows))

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-csv", type=Path, default=OUTPUT_DIR / "main_model" / "runs.csv")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <runs-csv dir>/ablation_tables")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.runs_csv.exists():
        raise SystemExit(f"runs.csv not found: {args.runs_csv} - run mainmodel_training first")
    out_dir = args.out_dir or args.runs_csv.parent / "ablation_tables"
    written = run(args.runs_csv, out_dir)
    print(f"wrote {len(written)} tables to {out_dir}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
