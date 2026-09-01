# Pipeline

End-to-end reproduction of PHYSHADE-Net, from source archives to building-height
estimates. **A CUDA GPU is assumed for the training stages** (as in the thesis); the
data-build and geometry stages run on CPU.

Nothing here ships with the repo: the imagery, annotations and weights are all rebuilt
locally. Paths default to `<repo>/Dataset`, `<repo>/Output`, `<repo>/data_sources`
(override the repo root with `PHYSHADE_ROOT`).

---

## 0. `data_sources/` — the canonical staging tree (git-ignored)

```
data_sources/
├── PHYSHADE_Image_Pairs.zip     52 raw geotiffs  (data_{summer,winter}_locN_M.tif.tiff)
├── proper_annotations_1.tar     Supervisely export 2025-04-23 20-49-28  (classes: Building / Shadow from Building)
├── proper_annotations_2.tar     Supervisely export 2025-04-23 19-28-48  (classes: Buildings / Shadows from buildings)
├── LR_BMO_2023.zip              summer flight metadata → 2023_beeldmiddenoverzicht_lrl.shp  (OPNAMEDATU / OPNAMETIJD)
├── BMO2023.zip                  winter flight metadata → 2023_beeldmiddenoverzicht_hrl.shp  (fotodatum / fototyd)
├── luo_validation.tar           Luo et al. cross-domain comparison export
├── val_source.zip               held-out val-set imagery, footprints, annotations  (~15 MB)
└── AISD/
    ├── Train412/{shadow,mask}   412 + 412 .tif   (1 = sunlit, 0 = shadow)
    ├── Val51/{shadow,mask}       51 +  51 .tif
    └── Test51/{shadow,mask}      51 +  51 .tif
```

The two annotation exports are **disjoint by tile** and together cover all 52 tiles;
`proper_annotations_1.tar` additionally carries the `masks_machine/` class rasters.

---

## Run everything

```powershell
./scripts/run_pipeline.ps1 -Smoke         # PowerShell: A + B at 2 epochs / 6 configs, height on annotated shadows
./scripts/run_pipeline.ps1                # thesis scale (all 26 configs, default epochs)
./scripts/run_pipeline.ps1 -SkipTrain     # data build + evaluation on an existing Output/
./scripts/run_pipeline.ps1 -Bag 3dbag.gpkg -DtmDir .\ahn_dtm   # also rebuild the height ground truth
```
```bash
scripts/run_pipeline.sh --smoke           # bash: same flags as --smoke / --skip-train / --bag / --dtm-dir
```

The stages below are what it runs, in order; each is also a standalone
`python -m physhade.<stage>` (or `physhade-*`) CLI.

---

## A. Data build

| # | command | produces |
|---|---|---|
| A1 | `python -m physhade.data.build_dataset` | `Dataset/main_model/all/` (raw, image, class_masks, building_footprint, annotated_shadow, annotation_export, human/accepted), `Dataset/val_set/`, `Dataset/aisd/`, `Dataset/main_model/shpfiles/` |
| A2 | `python -m physhade.data.training_preprocessing` | per-tile solar geometry → `Dataset/main_model/all/smear_shadow/`, then `Dataset/main_model/train/{image,smear_shadow,annotated_shadow}` + `pairs.txt` / `singles.txt` |
| A3 | `python -m physhade.data.build_base_dataset` | `Dataset/base_model/{train,val}/{shadow,mask}` from AISD, **mask polarity flipped** so `shadow = 1` |
| A4 | `python -m physhade.data.stage_luo` | `Dataset/main_model/luo/{img,masks_machine}` for the cross-domain qualitative comparison |

"Accepted" tiles = those whose class mask is non-empty (the thesis "38 of 52" filter);
the resulting list is pinned in `src/physhade/data/accepted_tiles.txt`.

---

## B. Model pipeline (CUDA GPU)

| # | command | notes |
|---|---|---|
| B1 | `python -m physhade.train.basemodel_training --data-dir Dataset/base_model` | RGB U-Net pretraining on AISD → `Output/base_model/<ts>/best_model.pth` |
| B2 | `python -m physhade.train.mainmodel_training` | 5-fold CV ablation; `--base-checkpoint` auto-discovers the newest B1 run; `--configs` selects loss configs |
| B3 | `python -m physhade.train.final_training` | trains the single final model on all folds + val_set |
| B4 | `python -m physhade.inference.quick_inference` / `physhade-infer` | metrics CSV + lineup panels |
| B5 | `python -m physhade.inference.qual_analysis` (reads `runs.csv`) or `.qual_final_model --checkpoint …` | qualitative strips + mosaics |
| B5c| `python -m physhade.inference.ablation_tables` / `physhade-ablation-tables` | the segmentation ablation stat tables (per-fold, per-subset paired t-tests + CIs, ranked summary) from `runs.csv` — no inference |
| B5d| `python -m physhade.inference.fold_complexity` / `physhade-fold-complexity` | per-fold edge density / shadow coverage / contrast + boxplots — explains fold-to-fold variance |
| B5e| `python -m physhade.inference.disagreement --masks-root … --pairs A:B,…` | RGB overlays of where two models' shadow predictions agree / disagree, + per-uid fraction CSV |
| B6 | `python -m physhade.height.height_pipeline --from annotated` | per-blob height from the annotated shadows, vs 3D BAG truth |
| B6b| `python -m physhade.height.height_pipeline --from checkpoint` | ...from one model's predicted shadows (`--checkpoint`, else newest final) |
| B6c| `python -m physhade.height.height_pipeline --from runs [--final]` | ...for **every `exp_id × fold`** in a `runs.csv` (add `--luo-baseline <ckpt>`) |
| B6d| `python -m physhade.height.height_pipeline --val --from runs --final` | the held-out val tiles, final models, vs `heights_validation_area.gpkg` |
| — | add `--diag` to any `height_pipeline` run | per-tile `<out>/diag/.../<uid>_{map,rays}.png` — labelled blob raster + RGB/match, and each blob's p99 ray with its shadow length |
| B6e| `python -m physhade.height.height_estimation` | synthetic-shadow control on the *old* raycaster (each tile's real azimuth) |
| B6f| `python -m physhade.height.height_estimation --sweep` | azimuth × ray-length-percentile sweep on the **shipped** method — reproduces the p99-vs-p75 ordering (thesis Table 16) |

> **Thesis method.** For each tile: `enforce_shadow_gap`
> (assign-and-break) relabels the shadow raster into per-building blobs separated by a
> 1 px gap; `subpixel_flood_shadow_height_vectorized` casts rays from each blob's
> *sun-facing edge pixels*, takes the **99th-percentile** ray length → per-blob height,
> and drops blobs whose edge touches the tile border (`truncated`). Each surviving blob
> is matched to a **3D BAG** building by marching its centroid toward the sun; the
> per-image footprints are grouped/dissolved by raster component and carry the largest
> member's `actual_height` (else `b3_h_70p - dtm_70p`).
>
> Outputs in `Output/main_model/height/<tag>_<ts>/`: `per_blob_metrics.csv` (thesis
> schema, one row per blob, `Is Highest Blob?` = the tallest blob per building),
> `errors.csv` (total / truncated blob counts), and blob-weighted
> `weighted_{per_image,model_image,model}_metrics.csv` (ME / MAE / RMSE / Std-resid,
> `AVERAGE_WEIGHTED` row, `LUO_UNET` excluded from the weighted mean). Metrics are
> reported over the accepted set (`true_height` present **and** `is_highest_blob`).
>
> The GeoPackage defaults to `data_sources/heights_training_area.gpkg`, or
> `heights_validation_area.gpkg` with `--val` (which also switches `--data-dir` to
> `Dataset/val_set`). `--no-bag` skips it, `--limit N` shortens a run, `--percentile`
> overrides 99, `--downsample K` decimates the masks (default 1 = 0.25 m grid; `2` is
> ~8x faster for smoke runs). `--from runs` runs inference per model first, writing
> `inferred_masks/<exp_id>/fold_<fold>/<ckpt>/`.

Every checkpoint-consuming stage takes `--checkpoint` / `--base-checkpoint`; with none
given it picks the most recent matching run under `Output/` and exits with an actionable
message if there is none.

---

## Known limitations
- **AISD polarity.** The AISD masks in `data_sources/AISD` encode `1 = sunlit`;
  `build_base_dataset` flips them (`--no-flip` to disable) and prints the shadow-pixel
  fraction per split as a sanity check.
- **Import order (Windows).** The three `src/physhade/train/*.py` modules `import rasterio`
  before `torchvision` on purpose due to a GDAL/torch DLL load-order clash.
- **`GDAL_DATA is not defined`** warnings on every geodata command are harmless (pyogrio
  still resolves the data files).
