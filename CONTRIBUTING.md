# Contributing

This is the research code for an MSc thesis, kept as a reproducible reference rather than
an actively maintained library. Issues and small PRs (bug fixes, docs, reproducibility)
are welcome, but may not actively be implemented.

## Setup

```bash
conda env create -f environment.yml
conda activate physhade
pre-commit install
```

## Before opening a PR

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

CI runs the same three checks. `pre-commit` runs `ruff` + whitespace hooks on commit.

## Conventions

- All pipeline stages are `python -m physhade.<stage>` argparse CLIs — no hardcoded
  paths, no work at import time. New stages follow the same `build_parser()` / `main()`
  / `run()` shape.
- Paths come from `physhade.config` (`DATA_DIR`, `OUTPUT_DIR`, `SOURCES_DIR`).
- Checkpoints are resolved via `config.resolve_checkpoint(...)`, never hardcoded.
- Data layout and the source→dataset build are documented in
  [`docs/pipeline.md`](docs/pipeline.md) and [`DATA.md`](DATA.md).
