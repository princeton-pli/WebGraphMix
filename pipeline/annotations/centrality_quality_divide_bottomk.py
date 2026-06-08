import glob
import io
import json
import os
import sys

import numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import default_scores_path, extract_hostname, get_data_root

corpus = get_data_root()
SCORES_PATH = str(default_scores_path("katz"))
DCLM_DIR = str(corpus / "scores_dclm-fasttext")
DOCS_DIR = str(corpus / "documents")
OUTPUT_ANNOTATIONS_DIR = str(
    corpus
    / "annotations_for_importance_based_document_sampling_bottomk"
    / "centrality_dclmfilter_divide_scores/katz"
)

POISON_PILL = 999.0
TEMP = 1.0

def get_global_dclm_max():
    print("Performing first pass to find global DCLM max...")
    dclm_files = glob.glob(os.path.join(DCLM_DIR, "*_processed.npy"))
    global_max = -float('inf')
    for f in dclm_files:
        try:
            arr_max = np.max(np.load(f))
            if arr_max > global_max: global_max = arr_max
        except Exception: continue
    print(f"Found Global DCLM Max: {global_max}")
    return global_max

def generate_annotations():
    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)

    with open(SCORES_PATH, 'r') as f:
        host_scores = json.load(f)

    max_graph_score = max(host_scores.values()) if host_scores else 1.0
    normalized_graph_scores = {k: np.exp((v - max_graph_score) / TEMP) for k, v in host_scores.items()}

    max_dclm_score = get_global_dclm_max()
    shard_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.jsonl.zst")))
    dctx = zstd.ZstdDecompressor()

    global_total_docs = 0
    global_matched_docs = 0
    global_unmapped_docs = 0
    unique_hosts_found = set()

    print(f"Processing {len(shard_files)} shards for Bottom-K...")

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path).replace(".jsonl.zst", "")
        output_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, f"{shard_name}.npy")
        dclm_path = os.path.join(DCLM_DIR, f"{shard_name}.npy")

        if os.path.exists(output_path) or not os.path.exists(dclm_path):
            continue

        dclm_scores = np.load(dclm_path)
        scores_for_shard = []

        with open(shard_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for idx, line in enumerate(text_stream):
                    global_total_docs += 1
                    raw_dclm = dclm_scores[idx]

                    doc = json.loads(line)
                    url = doc.get('url') or doc.get('metadata', {}).get('WARC-Target-URI')
                    host = extract_hostname(url)

                    if host not in normalized_graph_scores:
                        scores_for_shard.append(POISON_PILL)
                        global_unmapped_docs += 1
                    else:
                        norm_graph = normalized_graph_scores[host]
                        norm_dclm = np.exp((raw_dclm - max_dclm_score) / TEMP)
                        # Divide centrality by quality: bottom-k selects low centrality + high quality
                        scores_for_shard.append(norm_graph / norm_dclm)
                        global_matched_docs += 1
                        unique_hosts_found.add(host)

        np.save(output_path, np.array(scores_for_shard, dtype=np.float32))

    # --- FINAL STATISTICS ---
    print("\n" + "="*50)
    print("FINAL ALIGNMENT STATISTICS (DIVIDE BOTTOM-K)")
    print("="*50)
    print(f"Total Documents Processed:          {global_total_docs:,}")
    print(f"Docs Unmapped (Missing Host):       {global_unmapped_docs:,}")
    print(f"Docs Matched & Scored Successfully: {global_matched_docs:,}")
    print(f"Unique Hosts Found in Docs:         {len(unique_hosts_found):,}")
    print("="*50)

if __name__ == "__main__":
    generate_annotations()
