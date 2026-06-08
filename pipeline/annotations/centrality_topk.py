"""Generate centrality-only top-K annotation scores per document shard."""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import extract_hostname, get_centrality_results_dir, get_data_root


def parse_args():
    parser = argparse.ArgumentParser(description="Generate centrality-only top-K annotations.")
    parser.add_argument(
        "--scores-path",
        default=None,
        help="Host centrality JSON. Default: pipeline/graph/centrality/results/host_graph_scores_{metric}_*.json",
    )
    parser.add_argument(
        "--centrality-metric",
        default="betweenness",
        choices=["betweenness", "katz", "eigenvector"],
    )
    parser.add_argument("--corpus-root", default=None, help="Corpus-200B root (default: $DATA_ROOT).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output annotations dir (default: corpus_root/annotations_for_importance_based_document_sampling_topk/{metric}).",
    )
    return parser.parse_args()


def default_scores_path(metric: str) -> str:
    results_dir = get_centrality_results_dir()
    patterns = {
        "betweenness": "host_graph_scores_betweenness_k1400000.json",
        "katz": "host_graph_scores_katz_maxiter1000.json",
        "eigenvector": "host_graph_scores_eigenvector_maxiter1000.json",
    }
    return str(results_dir / patterns[metric])


def generate_annotations(scores_path: str, docs_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading host scores from {scores_path}...")
    with open(scores_path, "r") as f:
        host_scores = json.load(f)
    target_count = len(host_scores)

    shard_files = sorted(glob.glob(os.path.join(docs_dir, "*.jsonl.zst")))
    dctx = zstd.ZstdDecompressor()

    global_total_docs = 0
    global_matched_docs = 0
    unique_hosts_found = set()

    print(f"Starting processing of {len(shard_files)} shards...")

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path).replace(".jsonl.zst", "")
        output_path = os.path.join(output_dir, f"{shard_name}.npy")
        is_already_done = os.path.exists(output_path)
        scores_for_shard = []

        with open(shard_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    global_total_docs += 1
                    doc = json.loads(line)
                    url = doc.get("url") or doc.get("metadata", {}).get("WARC-Target-URI")

                    score = 0.0
                    if url:
                        host = extract_hostname(url)
                        if host in host_scores:
                            score = host_scores[host]
                            global_matched_docs += 1
                            unique_hosts_found.add(host)

                    if not is_already_done:
                        scores_for_shard.append(score)

        if not is_already_done:
            np.save(output_path, np.array(scores_for_shard, dtype=np.float32))

    print("\n" + "=" * 50)
    print("FINAL ALIGNMENT STATISTICS")
    print("=" * 50)
    print(f"Total Documents Processed:        {global_total_docs:,}")
    print(f"Total Documents Assigned a Score: {global_matched_docs:,}")
    print(f"Unique Hosts Found in Docs:       {len(unique_hosts_found):,}")
    print(f"Unique Hosts Expected from Graph: {target_count:,}")
    missing = target_count - len(unique_hosts_found)
    if missing == 0:
        print("SUCCESS: all graph hosts matched to corpus documents.")
    else:
        print(f"Note: {missing:,} graph hosts were not found in this pass.")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    corpus_root = args.corpus_root or str(get_data_root())
    metric = args.centrality_metric
    scores_path = args.scores_path or default_scores_path(metric)
    docs_dir = os.path.join(corpus_root, "documents")
    output_dir = args.output_dir or os.path.join(
        corpus_root,
        "annotations_for_importance_based_document_sampling_topk",
        metric,
    )
    generate_annotations(scores_path, docs_dir, output_dir)
