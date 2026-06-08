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
DOCS_DIR = str(corpus / "documents")
OUTPUT_ANNOTATIONS_DIR = str(
    corpus / "annotations_for_importance_based_document_sampling_bottomk/katz"
)

def generate_annotations():
    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)
    
    print("Loading host scores...")
    with open(SCORES_PATH, 'r') as f:
        host_scores = json.load(f)
    
    # --- LOGIC CHANGE: Determine "Poison Pill" score for unmapped URLs ---
    # We find the max score and use it for anyone not in the graph
    # so they are the LAST to be picked in a Bottom-K scenario.
    if host_scores:
        max_graph_score = max(host_scores.values())
        print(f"Max score found in graph: {max_graph_score}. Unmapped URLs will receive this score.")
    else:
        max_graph_score = 0.0
        print("Warning: host_scores is empty!")

    shard_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.jsonl.zst")))
    dctx = zstd.ZstdDecompressor()

    global_total_docs = 0
    global_matched_docs = 0
    unique_hosts_found = set()

    print(f"Processing {len(shard_files)} shards...")

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path).replace(".jsonl.zst", "")
        output_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, f"{shard_name}.npy")
        
        if os.path.exists(output_path):
            # We still iterate to collect stats, but skipping .npy write
            is_already_done = True 
        else:
            is_already_done = False
        
        scores_for_shard = []
        
        with open(shard_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    global_total_docs += 1
                    doc = json.loads(line)
                    url = doc.get('url') or doc.get('metadata', {}).get('WARC-Target-URI')
                    
                    # --- LOGIC CHANGE: Default to max_graph_score ---
                    score = max_graph_score 
                    
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

    print("\n" + "="*50)
    print("FINAL ALIGNMENT STATISTICS (BOTTOM-K READY)")
    print("="*50)
    print(f"Total Documents Processed:        {global_total_docs:,}")
    print(f"Total Documents Matched to Graph: {global_matched_docs:,}")
    print(f"Total Documents Unmapped (Max Score): {global_total_docs - global_matched_docs:,}")
    print("="*50)

if __name__ == "__main__":
    generate_annotations()