#!/usr/bin/env bash
# Run the whole PHYSHADE-Net pipeline end to end: data build -> training ->
# segmentation evaluation -> height estimation -> README figures.
#
# Usage:
#   scripts/run_pipeline.sh [--smoke] [--skip-train] [--bag FILE --dtm-dir DIR]
#
#   --smoke         2 epochs / 2 folds / 6 configs, height on annotated shadows
#                   (proves the pipeline runs; not thesis numbers). Default: full
#                   scale (all 26 configs, default epoch counts - hours to days on a GPU).
#   --skip-train    reuse whatever is already under Output/ (data build + eval only).
#   --bag / --dtm-dir   rebuild data_sources/heights_validation_area.gpkg from a 3D BAG
#                   GeoPackage + AHN DTM tiles; omitted -> keep the committed gpkg.
#
# Assumes the `physhade` env is active (`pip install -e .`) and data_sources/ is
# populated (see DATA.md). A CUDA GPU is assumed for the training stages.
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0 SKIP_TRAIN=0 BAG="" DTM_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --bag) BAG="${2:?}"; shift ;;
    --dtm-dir) DTM_DIR="${2:?}"; shift ;;
    -h|--help) sed -n '2,16p' "$0" | cut -c3-; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

PY="${PYTHON:-python}"
RUNS_CSV="Output/main_model/runs.csv"
HRUN="Output/main_model/height/pipeline_run"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; "$@"; }
have_runs() { [ -f "$RUNS_CSV" ]; }

if [ "$SMOKE" = 1 ]; then
  TRAIN_ARGS=(--epochs 2)
  MAIN_ARGS=(--configs BASE_BCE,BASE_DICE,RGB_BCE50_DICE50,RGBS_BCE50_DICE50,PHYS_ATT_0.5,HYB_DICE_PHYS10 --folds 2 --epochs 2)
  SWEEP_PCT="99,90,75"
else
  TRAIN_ARGS=()
  MAIN_ARGS=(--configs all)
  SWEEP_PCT="100,99,95,90,75"
fi

# ------------------------------------------------------------------ A. data build
step "$PY" -m physhade.data.build_dataset
step "$PY" -m physhade.data.training_preprocessing
step "$PY" -m physhade.data.training_preprocessing --val
step "$PY" -m physhade.data.build_base_dataset --overwrite
step "$PY" -m physhade.data.stage_luo

if [ -n "$BAG" ] && [ -n "$DTM_DIR" ]; then
  step "$PY" -m physhade.data.build_heights --bag "$BAG" --bag-layer lod12_2d \
    --dtm-dir "$DTM_DIR" --area validation --clip Dataset/val_set/building_footprint
else
  echo "==> skipping build_heights (pass --bag and --dtm-dir); using committed heights_*_area.gpkg"
fi

# --------------------------------------------------------------------- B. training
if [ "$SKIP_TRAIN" = 0 ]; then
  step "$PY" -m physhade.train.basemodel_training "${TRAIN_ARGS[@]}"
  step "$PY" -m physhade.train.mainmodel_training "${MAIN_ARGS[@]}"
  step "$PY" -m physhade.train.final_training "${TRAIN_ARGS[@]}"
fi

# ------------------------------------------------------ C. segmentation evaluation
step "$PY" -m physhade.inference.quick_inference
step "$PY" -m physhade.inference.fold_complexity
if have_runs; then
  step "$PY" -m physhade.inference.qual_analysis
  step "$PY" -m physhade.inference.ablation_tables
else
  echo "==> skipping qual_analysis / ablation_tables (no $RUNS_CSV)"
fi

# ---------------------------------------------------------------- D. height + figs
step "$PY" -m physhade.height.height_estimation --sweep --percentiles "$SWEEP_PCT"

rm -rf "$HRUN"
if [ "$SMOKE" = 0 ] && have_runs; then
  step "$PY" -m physhade.height.height_pipeline --val --from runs --final --diag --out-dir "$HRUN"
  step "$PY" -m physhade.inference.disagreement --masks-root "$HRUN/inferred_masks" \
    --pairs HYB_BCE_PHYS10:RGBS_BCE50_DICE50 || echo "  (disagreement skipped)"
else
  step "$PY" -m physhade.height.height_pipeline --val --from annotated --diag --out-dir "$HRUN"
fi

step "$PY" scripts/make_figures.py

printf '\n\033[1mpipeline finished.\033[0m  height results: %s/  figures: docs/figures/\n' "$HRUN"
