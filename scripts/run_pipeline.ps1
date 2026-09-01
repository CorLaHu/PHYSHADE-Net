<#
.SYNOPSIS
  Run the whole PHYSHADE-Net pipeline end to end.

.DESCRIPTION
  data build -> training -> segmentation evaluation -> height estimation -> README figures.
  Each stage is also a standalone `python -m physhade.<stage>` CLI (see docs/pipeline.md);
  checkpoint auto-discovery makes the ordering self-enforcing.

  Assumes the `physhade` env is active (`pip install -e .`) and data_sources/ is
  populated (see DATA.md). A CUDA GPU is assumed for the training stages.

.PARAMETER Smoke
  2 epochs / 2 folds / 6 configs, height on annotated shadows - proves the pipeline
  runs (not thesis numbers). Default: full scale (all 26 configs, default epoch counts;
  hours to days on a GPU).

.PARAMETER SkipTrain
  Reuse whatever is already under Output/ (data build + evaluation only).

.PARAMETER Bag
  3D BAG GeoPackage - rebuild data_sources/heights_validation_area.gpkg (needs -DtmDir).

.PARAMETER DtmDir
  Directory of AHN DTM raster tiles (*.tif / *.TIF), for -Bag.

.PARAMETER Python
  Python interpreter (default: python).

.EXAMPLE
  ./scripts/run_pipeline.ps1 -Smoke

.EXAMPLE
  ./scripts/run_pipeline.ps1 -Bag 3dbag.gpkg -DtmDir .\ahn_dtm
#>
[CmdletBinding()]
param(
    [switch] $Smoke,
    [switch] $SkipTrain,
    [string] $Bag = "",
    [string] $DtmDir = "",
    [string] $Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$RunsCsv = "Output/main_model/runs.csv"
$HRun = "Output/main_model/height/pipeline_run"

function Stage {
    param([Parameter(Mandatory)] [string[]] $Cmd)
    Write-Host "`n==> python $($Cmd -join ' ')" -ForegroundColor Cyan
    & $Python @Cmd
    if ($LASTEXITCODE -ne 0) { throw "stage failed (exit $LASTEXITCODE): python $($Cmd -join ' ')" }
}
function Test-Runs { Test-Path $RunsCsv }

if ($Smoke) {
    $TrainArgs = @("--epochs", "2")
    $MainArgs = @("--configs",
        "BASE_BCE,BASE_DICE,RGB_BCE50_DICE50,RGBS_BCE50_DICE50,PHYS_ATT_0.5,HYB_DICE_PHYS10",
        "--folds", "2", "--epochs", "2")
    $SweepPct = "99,90,75"
}
else {
    $TrainArgs = @()
    $MainArgs = @("--configs", "all")
    $SweepPct = "100,99,95,90,75"
}

# ---------------------------------------------------------------- A. data build
Stage "-m", "physhade.data.build_dataset"
Stage "-m", "physhade.data.training_preprocessing"
Stage "-m", "physhade.data.training_preprocessing", "--val"
Stage "-m", "physhade.data.build_base_dataset", "--overwrite"
Stage "-m", "physhade.data.stage_luo"

if ($Bag -and $DtmDir) {
    Stage (@("-m", "physhade.data.build_heights", "--bag", $Bag, "--bag-layer", "lod12_2d",
            "--dtm-dir", $DtmDir, "--area", "validation",
            "--clip", "Dataset/val_set/building_footprint"))
}
else {
    Write-Host "==> skipping build_heights (pass -Bag and -DtmDir); using committed heights_*_area.gpkg" -ForegroundColor Yellow
}

# ------------------------------------------------------------------ B. training
if (-not $SkipTrain) {
    Stage (@("-m", "physhade.train.basemodel_training") + $TrainArgs)
    Stage (@("-m", "physhade.train.mainmodel_training") + $MainArgs)
    Stage (@("-m", "physhade.train.final_training") + $TrainArgs)
}

# ------------------------------------------------- C. segmentation evaluation
Stage "-m", "physhade.inference.quick_inference"
Stage "-m", "physhade.inference.fold_complexity"
if (Test-Runs) {
    Stage "-m", "physhade.inference.qual_analysis"
    Stage "-m", "physhade.inference.ablation_tables"
}
else {
    Write-Host "==> skipping qual_analysis / ablation_tables (no $RunsCsv)" -ForegroundColor Yellow
}

# --------------------------------------------------------------- D. height + figs
Stage "-m", "physhade.height.height_estimation", "--sweep", "--percentiles", $SweepPct

if (Test-Path $HRun) { Remove-Item -Recurse -Force $HRun }
if ((-not $Smoke) -and (Test-Runs)) {
    Stage "-m", "physhade.height.height_pipeline", "--val", "--from", "runs", "--final", "--diag", "--out-dir", $HRun
    try {
        Stage "-m", "physhade.inference.disagreement", "--masks-root", "$HRun/inferred_masks",
        "--pairs", "HYB_BCE_PHYS10:RGBS_BCE50_DICE50"
    }
    catch { Write-Host "  (disagreement skipped: $_)" -ForegroundColor Yellow }
}
else {
    Stage "-m", "physhade.height.height_pipeline", "--val", "--from", "annotated", "--diag", "--out-dir", $HRun
}

Stage "scripts/make_figures.py"

Write-Host "`npipeline finished." -ForegroundColor Green
Write-Host "height results: $HRun/   figures: docs/figures/"
