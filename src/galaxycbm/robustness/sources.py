"""Cross-survey source registry + downloaders.

Access points verified against the HF Hub (`huggingface_hub` search):
    Euclid Q1:  https://huggingface.co/datasets/mwalmsley/gz_euclid
    JWST/HST:   https://huggingface.co/datasets/mwalmsley/gz_jwst_cosmos

Both are dataset repos alongside the primary `mwalmsley/gz_evo` used by
P1. Column layouts mirror gz_evo (id_str, image, ra, dec, per-survey vote
fractions) with survey-specific fraction-column suffixes.

The registry below is the default; override the URL / suffix per survey
in `configs/data.yaml → robustness` so the code never carries hard-coded
science choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SurveySource:
    name: str
    hf_repo: str
    suffix: str      # vote-fraction column suffix, e.g. "euclid", "hst_jwst"
    dataset_name: str  # value stored in the `dataset_name` column of the parquet
    citation: str


DEFAULT_SOURCES: dict[str, SurveySource] = {
    "euclid_q1": SurveySource(
        name="euclid_q1",
        hf_repo="mwalmsley/gz_euclid",
        suffix="euclid",
        dataset_name="gz_euclid",
        citation="Galaxy Zoo Euclid Q1 — Walmsley et al. (mwalmsley/gz_euclid)",
    ),
    "hst_jwst_cosmos_web": SurveySource(
        name="hst_jwst_cosmos_web",
        hf_repo="mwalmsley/gz_jwst_cosmos",
        suffix="jwst",
        dataset_name="gz_jwst_cosmos",
        citation="Galaxy Zoo JWST/COSMOS-Web — Walmsley et al. (mwalmsley/gz_jwst_cosmos)",
    ),
}


def download_all_shards(source: SurveySource, out_root: str | Path) -> list[Path]:
    """Snapshot every parquet under `data/<split>-*.parquet` from the HF repo.

    Uses huggingface_hub.snapshot_download with a strict allow_patterns filter
    so we never pull the entire repo (which may include large image caches).
    """
    from huggingface_hub import snapshot_download

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=source.hf_repo,
        repo_type="dataset",
        local_dir=out_root,
        allow_patterns=["data/*.parquet", "tiny/*.parquet"],
    )
    return sorted(Path(local).rglob("*.parquet"))
