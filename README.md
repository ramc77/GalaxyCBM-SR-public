# GalaxyCBM-SR

A **glass-box galaxy morphology classifier**: a Concept Bottleneck Model whose
dense linear head is replaced by a **symbolic decision rule** discovered by
PySR, wrapped in **split-conformal prediction**, and stress-tested across
DECaLS/SDSS → Euclid → HST/JWST domain shift.

Every number reported in the accompanying paper is recorded in
`results/metrics.json`. `paper/claims.yaml` lists the paper's claims, and
`galaxycbm.eval.check_claims` checks them against that file.

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
```

## Layout

| Path | Contents |
|------|----------|
| `src/galaxycbm/data` | Cutout cache, statmorph concepts, Hubble-type labels |
| `src/galaxycbm/concepts` | Stage 1: concept predictor (Zoobot backbone) |
| `src/galaxycbm/symbolic` | Stage 2: PySR symbolic head, rule export, frontier checks |
| `src/galaxycbm/uncertainty` | Stage 3: split-conformal and class-conditional calibration |
| `src/galaxycbm/baselines` | End-to-end CNN, dense linear CBM, XGBoost |
| `src/galaxycbm/robustness` | Cross-survey evaluation and rule stability |
| `src/galaxycbm/eval` | Metrics aggregation, figures, tables, claims check |
| `configs/` | Data, concept, model, symbolic and conformal settings |
| `results/` | Released results of each stage, with `run.json` records |
| `paper/` | Manuscript sources, figures, tables and the claims manifest |
| `tests/` | One test file per module |

## Reproducibility

- Python 3.11 pinned via `uv`.
- All third-party versions pinned in `pyproject.toml`; PySR's Julia is pinned
  by its bundled `juliapkg.json` and lives inside the venv, not the system.
- `galaxycbm.utils.runlog.write_run_json(stage_dir)` writes git SHA, package
  versions, config hash, and seed to `results/<stage>/run.json` for every
  stage run.
- Every `src` module has a `tests/` file; `uv run pytest` gates every stage.
