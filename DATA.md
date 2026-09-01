# Data & Provenance

The dataset is **not distributed in this repository**, but the full `Dataset/` tree is rebuilt
from the source archives with `pip install -e ".[build]"` and:

```bash
python -m physhade.data.build_dataset               # A1: main_model/all + val_set + aisd + shpfiles
python -m physhade.data.training_preprocessing       # A2: solar geometry -> smear priors -> pairs/singles
python -m physhade.data.training_preprocessing --val # A2v: regenerate Dataset/val_set/smear_shadow
python -m physhade.data.build_base_dataset           # A3: AISD -> base_model (polarity-flipped)
python -m physhade.data.stage_luo                    # A4: Luo et al. comparison tiles
```

(`build_dataset` already runs the `--val` step; run it standalone to refresh
just the val smear priors.)

The source archives live in a flat, git-ignored `data_sources/` directory; its
exact contents are documented in **[docs/pipeline.md](docs/pipeline.md)**.

`build_dataset` produces:

```
Dataset/
├── main_model/all/
│   ├── raw/                  52 × 512×512 geotiffs (PDOK Luchtfoto 25 cm, summer + winter)
│   ├── image/                georeferenced copies of raw/
│   ├── class_masks/          0 = background, 1 = building, 2 = shadow (from the exports' masks_machine)
│   ├── building_footprint/   binary building masks (per-export .png + split .tif)
│   ├── annotated_shadow/     binary shadow masks (38 of 52 tiles carry labels)
│   ├── smear_shadow/         physics priors (written by training_preprocessing)
│   ├── human/accepted/       the non-empty ("accepted") tile set — see accepted_tiles.txt
│   └── annotation_export/    verbatim Supervisely exports, for provenance
├── val_set/                  held-out tiles: georef_image/, annotated_shadows/, smear_shadow/, ann/, summary/
├── aisd/{Train412,Val51,Test51}/{shadow,mask}   cross-domain shadow set
└── base_model/{train,val}/{shadow,mask}         AISD pretraining split (A3)
```

`training_preprocessing` then tiles/filters `all/` into
`Dataset/main_model/train/{image,smear_shadow,annotated_shadow}` + `pairs.txt` /
`singles.txt`.

## Source data (public Dutch open geodata)

| Source | Role | Licence |
|---|---|---|
| PDOK Luchtfoto (aerial imagery, 25 cm GSD) | RGB input tiles | CC BY 4.0 (PDOK/Kadaster) |
| BGT (building footprints) | Smear-prior generation & raycasting base | CC BY 4.0 (Kadaster) |
| 3D BAG (LoD1.2/2.2 building models, BAG+AHN fusion) | Height percentiles for smear bounds; ground-truth heights | CC BY 4.0 (3D BAG project) |
| AHN (national point cloud/DTM) | True above-ground height sampling | CC BY 4.0 (Nationaal Georegister) |

## Annotations

Shadow and building annotations were drawn manually in **Supervisely**
(`Buildings` / `Shadows from buildings` bitmap classes; one export uses the
singular spellings `Building` / `Shadow from Building`) following written
guidelines (annotation protocol in the thesis appendices): buildings first,
only visible parts, "underestimate rather than overestimate" precision rule,
followed by a QA review pass. The dataset was annotated in two complementary
exports (dataset 2025-04-23 19-28-48 and dataset 2025-04-23 20-49-28), which
together cover all 52 tiles. Each export carries the class-encoded
`masks_machine/*.png` masks (0 = background, 1 = building, 2 = shadow) that
the original pipeline split into the binary layers; the builder reproduces
that split and falls back to decoding the `ann/*.json` bitmaps
(base64 → zlib → PNG) if the pngs are absent. 38 tiles carry labels; the
remainder are intentionally empty (no buildings/shadows), matching the
thesis' tile-filtering protocol.

Regenerable derived layers (smear priors, height ground truth) are produced
from the sources above; the six validation smear priors also ship inside
`val_source.zip`.

## Ground-truth building heights

`data_sources/heights_{training,validation}_area.gpkg` (layer `output_heights`)
are the reference ground truth the height pipeline scores against
(`height_pipeline.load_bag_heights`). Rebuild them with:

```bash
python -m physhade.data.build_heights \
    --bag 3dbag.gpkg --bag-layer lod12_2d \
    --dtm-dir ./ahn_dtm --area validation \
    --clip Dataset/val_set/building_footprint
```

Raw inputs:

| Source | Where | Attribute used |
|---|---|---|
| 3D BAG GeoPackage | [3dbag.nl](https://3dbag.nl) download (`lod12_2d` layer) | `b3_h_70p` — 70th-percentile roof height above NAP; `identificatie` |
| AHN DTM raster tiles | PDOK AHN (0.5 m grid, EPSG:28992, e.g. `M_*.TIF`) | terrain surface |

For each building: `dtm_70p` = 70th percentile of the DTM over the footprint
bounding box extended by 5 m; `actual_height = b3_h_70p - dtm_70p`. `--clip`
restricts the BAG to polygons intersecting an area of interest (a GeoPackage,
or a directory of georeferenced footprint rasters).

If you reuse the pipeline, download sources directly from the providers above
and follow their current terms; the extracted tiles should be regenerated
locally rather than redistributed.
