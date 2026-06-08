"""
Score Distribution Analysis
- Distribution of betweenness, eigenvector, katz scores
- High/low scoring hosts
- Example documents from those hosts
"""

import json
import os
import glob
import zstandard as zstd
import io
from urllib.parse import urlparse
from collections import defaultdict
import numpy as np
import multiprocessing as mp

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from paths import centrality_results_dir, data_root  # noqa: E402

RESULTS_DIR = centrality_results_dir()
DOCS_DIR = data_root() / "documents"

# ─── 1. Load Scores ──────────────────────────────────────────────────────────

print("Loading scores...")
scores = {}
score_files = {
    "betweenness": "host_graph_scores_betweenness_k1400000.json",
    "eigenvector": "host_graph_scores_eigenvector_maxiter1000.json",
    "katz":        "host_graph_scores_katz_maxiter1000.json",
}
for metric, fname in score_files.items():
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        scores[metric] = json.load(f)
    print(f"  {metric}: {len(scores[metric]):,} hosts loaded")

# ─── 2. Distribution Stats ───────────────────────────────────────────────────

print("\n=== Score Distributions ===")
for metric, d in scores.items():
    vals = np.array(list(d.values()), dtype=np.float64)
    nonzero = vals[vals > 0]
    print(f"\n[{metric}]")
    print(f"  total hosts  : {len(vals):,}")
    print(f"  zero scores  : {(vals == 0).sum():,} ({100*(vals==0).mean():.1f}%)")
    print(f"  non-zero     : {len(nonzero):,}")
    print(f"  min          : {vals.min():.3e}")
    print(f"  max          : {vals.max():.3e}")
    print(f"  mean         : {vals.mean():.3e}")
    print(f"  median       : {np.median(vals):.3e}")
    pcts = [50, 75, 90, 95, 99, 99.9, 99.99]
    for p in pcts:
        print(f"  p{p:<5}       : {np.percentile(vals, p):.3e}")

# ─── 3. Top / Bottom Hosts ───────────────────────────────────────────────────

N_SHOW = 30

print("\n=== Top & Bottom Hosts by Metric ===")
top_hosts = {}
bottom_hosts = {}

for metric, d in scores.items():
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:N_SHOW]
    # bottom: skip exact zeros for more informative results; show lowest non-zero
    nonzero_items = [(h, v) for h, v in sorted_items if v > 0]
    bottom = nonzero_items[-N_SHOW:][::-1]  # lowest non-zero, highest first in the list

    top_hosts[metric] = top
    bottom_hosts[metric] = bottom

    print(f"\n--- {metric} TOP {N_SHOW} ---")
    for rank, (host, score) in enumerate(top, 1):
        print(f"  {rank:>3}. {score:.4e}  {host}")

    print(f"\n--- {metric} BOTTOM {N_SHOW} (lowest non-zero) ---")
    for rank, (host, score) in enumerate(bottom, 1):
        print(f"  {rank:>3}. {score:.4e}  {host}")

# ─── 4. Collect Example Documents ────────────────────────────────────────────

# Build set of interesting hosts to look for
interesting_hosts = set()
for metric in scores:
    for host, _ in top_hosts[metric][:10]:
        interesting_hosts.add(host)
    for host, _ in bottom_hosts[metric][:10]:
        interesting_hosts.add(host)

EXAMPLES_PER_HOST = 3
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

# Free the large score dicts before forking workers to avoid CoW memory explosion
metric_names = list(scores.keys())
del scores
import gc; gc.collect()

def scan_shard(fpath):
    # target_hosts is a module-level global set after Pool is created via initializer
    results = {}
    dctx = zstd.ZstdDecompressor()
    try:
        with open(fpath, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    url = doc.get("url", "")
                    h = urlparse(url).hostname or ""
                    if h.startswith("www."):
                        h = h[4:]
                    if h in _TARGET_HOSTS:
                        results.setdefault(h, [])
                        if len(results[h]) < EXAMPLES_PER_HOST:
                            snippet = doc.get("text", "")[:400].replace("\n", " ")
                            results[h].append((url, snippet))
    except Exception:
        pass
    return results

_TARGET_HOSTS = frozenset(interesting_hosts)

def _pool_init(target):
    global _TARGET_HOSTS
    _TARGET_HOSTS = target

all_doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.jsonl.zst")))
print(f"\n=== Finding Example Documents for {len(interesting_hosts)} hosts ===")
print(f"(scanning all {len(all_doc_files)} shards with {N_WORKERS} workers)")

host_examples = defaultdict(list)
frozen_hosts = frozenset(interesting_hosts)

with mp.Pool(N_WORKERS, initializer=_pool_init, initargs=(frozen_hosts,)) as pool:
    for i, shard_result in enumerate(pool.imap_unordered(scan_shard, all_doc_files, chunksize=5)):
        for host, docs in shard_result.items():
            for doc in docs:
                if len(host_examples[host]) < EXAMPLES_PER_HOST:
                    host_examples[host].append(doc)
        if (i + 1) % 1000 == 0:
            found = sum(1 for h in interesting_hosts if host_examples[h])
            print(f"  processed {i+1}/{len(all_doc_files)} shards, {found}/{len(interesting_hosts)} hosts found so far...", flush=True)

found_count = sum(1 for h in interesting_hosts if host_examples[h])
missing = sorted(h for h in interesting_hosts if not host_examples[h])
print(f"  Found examples for {found_count}/{len(interesting_hosts)} hosts.")
if missing:
    print(f"  Still no examples for: {missing}")

# ─── 5. Print Examples ───────────────────────────────────────────────────────

print("\n=== Example Documents by Host ===")
for metric in metric_names:
    print(f"\n{'='*60}")
    print(f"  METRIC: {metric}")
    print(f"{'='*60}")

    print(f"\n  [ TOP HOSTS ]")
    for host, score in top_hosts[metric][:10]:
        examples = host_examples.get(host, [])
        print(f"\n  {host}  (score={score:.4e})")
        if examples:
            for url, snippet in examples:
                print(f"    URL    : {url}")
                print(f"    Snippet: {snippet[:200]}")
        else:
            print(f"    (no examples found in scanned shards)")

    print(f"\n  [ BOTTOM HOSTS ]")
    for host, score in bottom_hosts[metric][:10]:
        examples = host_examples.get(host, [])
        print(f"\n  {host}  (score={score:.4e})")
        if examples:
            for url, snippet in examples:
                print(f"    URL    : {url}")
                print(f"    Snippet: {snippet[:200]}")
        else:
            print(f"    (no examples found in scanned shards)")

print("\nDone.")
