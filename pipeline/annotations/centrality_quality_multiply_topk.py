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
SCORES_PATH = str(default_scores_path("betweenness"))
DCLM_DIR = str(corpus / "scores_dclm-fasttext")
DOCS_DIR = str(corpus / "documents")
OUTPUT_ANNOTATIONS_DIR = str(
    corpus
    / "annotations_for_importance_based_document_sampling_topk"
    / "centrality_dclmfilter_multiply_scores/betweenness"
)

QUALITY_THRESHOLD = -0.000003
TEMP = 1.0

def get_global_dclm_max():
    """First pass: Scan all DCLM .npy files to find the absolute highest score."""
    print("Performing first pass to find global DCLM max...")
    dclm_files = glob.glob(os.path.join(DCLM_DIR, "*.npy"))
    global_max = -float('inf')
    
    for f in dclm_files:
        try:
            arr_max = np.max(np.load(f))
            if arr_max > global_max:
                global_max = arr_max
        except Exception as e:
            print(f"Error reading {f} for max calculation: {e}")
            
    print(f"Found Global DCLM Max: {global_max}")
    return global_max

def generate_annotations():
    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)
    
    # --- 1. PREP GRAPH SCORES ---
    print("Loading host scores...")
    with open(SCORES_PATH, 'r') as f:
        host_scores = json.load(f)
    
    target_count = len(host_scores)
    
    # Find Max Graph Score & Pre-normalize to save compute time in the loop
    max_graph_score = max(host_scores.values()) if host_scores else 1.0
    print(f"Found Global Graph Max: {max_graph_score}")
    
    normalized_graph_scores = {}
    for host, score in host_scores.items():
        # Exponential normalization for graph scores
        normalized_graph_scores[host] = np.exp((score - max_graph_score) / TEMP)

    # --- 2. PREP DCLM SCORES ---
    max_dclm_score = get_global_dclm_max()

    # --- 3. MAIN ALIGNMENT LOOP ---
    shard_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.jsonl.zst")))
    dctx = zstd.ZstdDecompressor()

    global_total_docs = 0
    global_matched_docs = 0
    global_dropped_quality = 0
    unique_hosts_found = set()

    print(f"\nStarting processing of {len(shard_files)} shards. This may take a while...")

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path).replace(".jsonl.zst", "")
        output_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, f"{shard_name}.npy")
        dclm_path = os.path.join(DCLM_DIR, f"{shard_name}.npy")
        
        if os.path.exists(output_path):
            print(f"Skipping {shard_name}, output already exists.")
            continue
            
        if not os.path.exists(dclm_path):
            print(f"Skipping {shard_name}, matching DCLM score file not found.")
            continue

        # Load DCLM scores for this specific shard
        dclm_scores = np.load(dclm_path)
        scores_for_shard = []
        
        with open(shard_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for idx, line in enumerate(text_stream):
                    global_total_docs += 1
                    
                    # Gatekeeper 1: DCLM Quality Check
                    raw_dclm = dclm_scores[idx]
                    if raw_dclm <= QUALITY_THRESHOLD:
                        scores_for_shard.append(0.0)
                        global_dropped_quality += 1
                        continue
                        
                    # Gatekeeper 2: Graph Presence Check
                    doc = json.loads(line)
                    url = doc.get('url') or doc.get('metadata', {}).get('WARC-Target-URI')
                    
                    final_score = 0.0
                    if url:
                        host = extract_hostname(url)
                        if host in normalized_graph_scores:
                            # Both conditions met! Calculate combined score.
                            norm_graph = normalized_graph_scores[host]
                            norm_dclm = np.exp((raw_dclm - max_dclm_score) / TEMP)
                            
                            final_score = norm_graph * norm_dclm
                            
                            global_matched_docs += 1
                            unique_hosts_found.add(host)
                    
                    scores_for_shard.append(final_score)
        
        np.save(output_path, np.array(scores_for_shard, dtype=np.float32))

    # --- 4. PRINT STATS ---
    print("\n" + "="*50)
    print("FINAL ALIGNMENT STATISTICS (COMBINED TOP-K)")
    print("="*50)
    print(f"Total Documents Processed:          {global_total_docs:,}")
    print(f"Docs Dropped (Below DCLM Threshold): {global_dropped_quality:,}")
    print(f"Docs Matched & Scored Successfully: {global_matched_docs:,}")
    print(f"Unique Hosts Found in Docs:         {len(unique_hosts_found):,}")
    print("="*50)

if __name__ == "__main__":
    generate_annotations()