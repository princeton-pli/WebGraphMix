"""
doc_counts_by_host.py

1. Scan all corpus shards -> count docs AND tokens per host
   (reads jsonl.zst for URLs + aligned .npy for token counts)
2. Save host_doc_counts.json and host_token_counts.json for reuse
3. Load centrality scores (betweenness, eigenvector, katz)
4. For each metric, report docs and tokens in top/bottom 1/5/10/25% hosts
"""

import json
import os
import glob
import io
import multiprocessing as mp
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
import zstandard as zstd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from paths import centrality_dir, centrality_results_dir, data_root  # noqa: E402

DOCS_DIR = data_root() / "documents"
TOKENS_DIR = data_root() / "tokens"
RESULTS_DIR = centrality_results_dir()
OUTPUT_DIR = centrality_dir()
DOC_COUNTS_FILE   = os.path.join(OUTPUT_DIR, "host_doc_counts.json")
TOKEN_COUNTS_FILE = os.path.join(OUTPUT_DIR, "host_token_counts.json")
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))


# --- 1. Count docs and tokens per host ---------------------------------------

def count_shard(doc_path):
    """Returns (doc_counts, token_counts) dicts for one shard."""
    shard_name = os.path.basename(doc_path).replace(".jsonl.zst", "")
    token_path = os.path.join(TOKENS_DIR, shard_name + ".npy")

    doc_counts   = defaultdict(int)
    token_counts = defaultdict(int)

    try:
        token_arr = np.load(token_path)
    except Exception:
        token_arr = None

    dctx = zstd.ZstdDecompressor()
    try:
        with open(doc_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                doc_idx = 0
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
                    if not h:
                        doc_idx += 1
                        continue
                    doc_counts[h] += 1
                    if token_arr is not None and doc_idx < len(token_arr):
                        token_counts[h] += int(token_arr[doc_idx])
                    doc_idx += 1
    except Exception:
        pass

    return dict(doc_counts), dict(token_counts)


already_have_docs   = os.path.exists(DOC_COUNTS_FILE)
already_have_tokens = os.path.exists(TOKEN_COUNTS_FILE)

if already_have_docs and already_have_tokens:
    print("Loading existing counts...")
    with open(DOC_COUNTS_FILE) as f:
        host_doc_counts = json.load(f)
    with open(TOKEN_COUNTS_FILE) as f:
        host_token_counts = json.load(f)
    total_docs   = sum(host_doc_counts.values())
    total_tokens = sum(host_token_counts.values())
    print(f"  {len(host_doc_counts):,} hosts | {total_docs:,} docs | {total_tokens:,} tokens")
else:
    all_shards = sorted(glob.glob(os.path.join(DOCS_DIR, "*.jsonl.zst")))
    print(f"Scanning {len(all_shards)} shards with {N_WORKERS} workers...")

    host_doc_counts   = defaultdict(int)
    host_token_counts = defaultdict(int)

    with mp.Pool(N_WORKERS) as pool:
        for i, (shard_docs, shard_tokens) in enumerate(
            pool.imap_unordered(count_shard, all_shards, chunksize=5)
        ):
            for h, c in shard_docs.items():
                host_doc_counts[h] += c
            for h, c in shard_tokens.items():
                host_token_counts[h] += c
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(all_shards)} shards done...", flush=True)

    host_doc_counts   = dict(host_doc_counts)
    host_token_counts = dict(host_token_counts)
    total_docs   = sum(host_doc_counts.values())
    total_tokens = sum(host_token_counts.values())

    print(f"Done. {len(host_doc_counts):,} unique hosts | {total_docs:,} docs | {total_tokens:,} tokens")

    with open(DOC_COUNTS_FILE, "w") as f:
        json.dump(host_doc_counts, f)
    with open(TOKEN_COUNTS_FILE, "w") as f:
        json.dump(host_token_counts, f)
    print(f"Saved to {OUTPUT_DIR}")


# --- 2. Sanity check: top 20 hosts by docs and by tokens ---------------------

print("\n=== Top 20 Hosts by Document Count ===")
for rank, (host, cnt) in enumerate(
    sorted(host_doc_counts.items(), key=lambda x: x[1], reverse=True)[:20], 1
):
    tok = host_token_counts.get(host, 0)
    print(f"  {rank:>3}. {cnt:>8,} docs  {tok:>12,} tokens  {host}")

print("\n=== Top 20 Hosts by Token Count ===")
for rank, (host, cnt) in enumerate(
    sorted(host_token_counts.items(), key=lambda x: x[1], reverse=True)[:20], 1
):
    docs = host_doc_counts.get(host, 0)
    avg  = cnt / docs if docs else 0
    print(f"  {rank:>3}. {cnt:>12,} tokens  {docs:>8,} docs  avg {avg:>6.0f} tok/doc  {host}")


# --- 3. Load centrality scores -----------------------------------------------

score_files = {
    "betweenness": "host_graph_scores_betweenness_k1400000.json",
    "eigenvector": "host_graph_scores_eigenvector_maxiter1000.json",
    "katz":        "host_graph_scores_katz_maxiter1000.json",
}

print("\nLoading centrality scores...")
scores = {}
for metric, fname in score_files.items():
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        scores[metric] = json.load(f)
    print(f"  {metric}: {len(scores[metric]):,} hosts")


# --- 4. Percentile bucket analysis -------------------------------------------

PERCENTILES = [1, 5, 10, 25]

print(f"\nCorpus totals: {total_docs:,} docs | {total_tokens:,} tokens")
print(f"Mean tokens/doc (corpus-wide): {total_tokens/total_docs:.1f}")

for metric, score_dict in scores.items():
    print(f"\n{'='*80}")
    print(f"  METRIC: {metric}")
    print(f"{'='*80}")

    common_hosts = [h for h in score_dict if h in host_doc_counts]
    score_arr = np.array([score_dict[h] for h in common_hosts], dtype=np.float64)
    doc_arr   = np.array([host_doc_counts[h] for h in common_hosts], dtype=np.int64)
    tok_arr   = np.array([host_token_counts.get(h, 0) for h in common_hosts], dtype=np.int64)

    docs_no_score   = sum(host_doc_counts[h] for h in host_doc_counts if h not in score_dict)
    tokens_no_score = sum(host_token_counts.get(h, 0) for h in host_doc_counts if h not in score_dict)
    hosts_no_score  = sum(1 for h in host_doc_counts if h not in score_dict)

    total_scored_docs   = int(doc_arr.sum())
    total_scored_tokens = int(tok_arr.sum())

    print(f"  Hosts with score AND docs : {len(common_hosts):,}")
    print(f"  Hosts with score, no docs : {len(score_dict) - len(common_hosts):,}")
    print(f"  Hosts with docs, no score : {hosts_no_score:,}")
    print(f"    -> unscored docs   : {docs_no_score:,} ({100*docs_no_score/total_docs:.2f}% of corpus)")
    print(f"    -> unscored tokens : {tokens_no_score:,} ({100*tokens_no_score/total_tokens:.2f}% of corpus)")
    print(f"  Scored docs   : {total_scored_docs:,} ({100*total_scored_docs/total_docs:.2f}% of corpus)")
    print(f"  Scored tokens : {total_scored_tokens:,} ({100*total_scored_tokens/total_tokens:.2f}% of corpus)")

    sorted_idx = np.argsort(score_arr)  # ascending

    hdr = (f"  {'Group':<14} {'Hosts':>8}  {'Docs':>12}  "
           f"{'%Corp Docs':>11}  {'%Scrd Docs':>11}  "
           f"{'Tokens':>14}  {'%Corp Tok':>10}  {'%Scrd Tok':>10}")
    print(f"\n{hdr}")
    print(f"  {'-'*100}")

    for p in PERCENTILES:
        n_bucket = max(1, int(len(common_hosts) * p / 100))

        top_idx    = sorted_idx[-n_bucket:]
        top_docs   = int(doc_arr[top_idx].sum())
        top_tokens = int(tok_arr[top_idx].sum())

        bot_idx    = sorted_idx[:n_bucket]
        bot_docs   = int(doc_arr[bot_idx].sum())
        bot_tokens = int(tok_arr[bot_idx].sum())

        print(f"  TOP {p:>2}%       {n_bucket:>8,}  {top_docs:>12,}  "
              f"{100*top_docs/total_docs:>10.2f}%  {100*top_docs/total_scored_docs:>10.2f}%  "
              f"{top_tokens:>14,}  {100*top_tokens/total_tokens:>9.2f}%  {100*top_tokens/total_scored_tokens:>9.2f}%")
        print(f"  BOTTOM {p:>2}%    {n_bucket:>8,}  {bot_docs:>12,}  "
              f"{100*bot_docs/total_docs:>10.2f}%  {100*bot_docs/total_scored_docs:>10.2f}%  "
              f"{bot_tokens:>14,}  {100*bot_tokens/total_tokens:>9.2f}%  {100*bot_tokens/total_scored_tokens:>9.2f}%")
        print()

    # Weighted percentiles: what score does the average doc/token's host have?
    sorted_score = score_arr[sorted_idx]
    sorted_docs  = doc_arr[sorted_idx]
    sorted_toks  = tok_arr[sorted_idx]
    cum_docs     = np.cumsum(sorted_docs)
    cum_toks     = np.cumsum(sorted_toks)

    print(f"  Doc-weighted score percentiles (score of the host for the average doc):")
    for p in [1, 10, 25, 50, 75, 90, 99]:
        target = total_scored_docs * p / 100
        idx_p  = np.searchsorted(cum_docs, target)
        val    = sorted_score[min(idx_p, len(sorted_score) - 1)]
        print(f"    p{p:<3}: {val:.4e}")

    print(f"  Token-weighted score percentiles (score of the host for the average token):")
    for p in [1, 10, 25, 50, 75, 90, 99]:
        target = total_scored_tokens * p / 100
        idx_p  = np.searchsorted(cum_toks, target)
        val    = sorted_score[min(idx_p, len(sorted_score) - 1)]
        print(f"    p{p:<3}: {val:.4e}")

print("\nDone.")
