# GalaxyCBM-SR

A **glass-box galaxy morphology classifier**: a Concept Bottleneck Model whose
dense linear head is replaced by a **symbolic decision rule** discovered by
PySR, wrapped in **split-conformal prediction**, and stress-tested across
DECaLS/SDSS → Euclid → HST/JWST domain shift.

Every number reported in the accompanying paper traces to
`results/metrics.json` and is verified against `paper/claims.yaml` at build
time, so all published values regenerate from source.

## Quickstart

```bash
# 1. Install uv (once) and Julia is auto-managed by juliapkg on first `import pysr`.
brew install uv

# 2. Create the environment (Python 3.11).
#    Base install (scaffold + Stage 2/3 modelling, works on every platform):
uv sync --extra dev
#    Add Stage-1 modelling deps (torch, zoobot). Linux/aarch64/macOS-arm64
#    only — torch dropped macOS x86_64 wheels at 2.3, and zoobot needs 2.7+.
uv sync --extra dev --extra stage1

# 3. Smoke test — every module imports.
uv run pytest -q

# 4. Trigger the Julia install (first time only; downloads ~300 MB).
uv run python -c "import pysr; print(pysr.__version__)"

# 5. Full pipeline (once P1–P9 land).
make all
```

## Stage targets

Each `make` target corresponds to one pipeline stage:

| Target | Stage | Produces                                     |
|--------------------|--------|----------------------------------------------|
| `make data.raw`   | 1    | `data/raw/`, cutout manifest                 |
| `make data.concepts` | 2  | statmorph concepts merged into manifest      |
| `make data.labels` | 3    | `data/processed/dataset.parquet` + splits    |
| `make stage1`      | P4     | Concept predictor checkpoint + predictions   |
| `make stage2`      | P5     | PySR symbolic rules + rule table             |
| `make stage3`      | P6     | Conformal prediction sets + coverage report  |
| `make baselines`  | 7    | End-to-end CNN, dense CBM, XGBoost, SHAP     |
| `make robustness` | 8    | Cross-survey delta table                     |
| `make eval`       | 9    | Consolidated figures + `results/metrics.json`|
| `make all`         | —      | Everything above, in order                   |

## Reproducibility spine (P0)

- Python 3.11 pinned via `uv`.
- All third-party versions pinned in `pyproject.toml`; PySR's Julia is pinned
  by its bundled `juliapkg.json` and lives inside the venv, not the system.
- `galaxycbm.utils.runlog.write_run_json(stage_dir)` writes git SHA, package
  versions, config hash, and seed to `results/<stage>/run.json` for every
  stage run.
- Every `src` module has a `tests/` file; `uv run pytest` gates every stage.
