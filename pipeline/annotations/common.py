"""Shared utilities for WebGraphMix annotation generation."""
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Allow imports from lib/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from paths import centrality_results_dir, data_root, repo_root  # noqa: E402


def get_data_root() -> Path:
    return data_root()


def get_repo_root() -> Path:
    return repo_root()


def get_centrality_results_dir() -> Path:
    return centrality_results_dir()


def default_scores_path(metric: str) -> Path:
    patterns = {
        "betweenness": "host_graph_scores_betweenness_k1400000.json",
        "katz": "host_graph_scores_katz_maxiter1000.json",
        "eigenvector": "host_graph_scores_eigenvector_maxiter1000.json",
    }
    return centrality_results_dir() / patterns[metric]


def extract_hostname(url: str | None) -> str | None:
    if not url:
        return None
    try:
        hostname = urlparse(url).netloc
        if not hostname:
            return None
        if ":" in hostname:
            hostname = hostname.split(":")[0]
        if hostname.startswith("www."):
            return hostname[4:]
        return hostname
    except Exception:
        return None
