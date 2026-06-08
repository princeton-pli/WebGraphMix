"""Central path configuration for WebGraphMix."""
import os
from pathlib import Path


def repo_root() -> Path:
    return Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent)).resolve()


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", repo_root() / "corpus_200b")).resolve()


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).resolve()


def graph_data_dir() -> Path:
    return repo_root() / "pipeline" / "graph" / "data" / "commoncrawl"


def centrality_results_dir() -> Path:
    return repo_root() / "pipeline" / "graph" / "centrality" / "results"


def centrality_dir() -> Path:
    return repo_root() / "pipeline" / "graph" / "centrality"


def dclm_dir() -> Path:
    return repo_root() / "dclm"
