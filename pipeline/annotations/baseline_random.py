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
DOCS_DIR = str(corpus / "documents")
OUTPUT_ANNOTATIONS_DIR = str(corpus / "annotations/baseline_random")

def generate_annotations():
    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)

    print("Loading host graph to determine which hosts appear in graph...")
    with open(SCORES_PATH, 'r') as f:
        host_scores = json.load(f)
    graph_hosts = set(host_scores.keys())
    target_count = len(graph_hosts)

    shard_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.jsonl.zst")))
    dctx = zstd.ZstdDecompressor()

    global_total_docs = 0
    global_matched_docs = 0
    unique_hosts_found = set()

    print(f"Starting processing of {len(shard_files)} shards. This may take a while...")

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path).replace(".jsonl.zst", "")
        output_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, f"{shard_name}.npy")

        is_already_done = os.path.exists(output_path)

        scores_for_shard = []

        with open(shard_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    global_total_docs += 1
                    doc = json.loads(line)
                    url = doc.get('url') or doc.get('metadata', {}).get('WARC-Target-URI')

                    score = 0.0
                    if url:
                        host = extract_hostname(url)
                        if host in graph_hosts:
                            score = np.random.uniform(1.0, 2.0)
                            global_matched_docs += 1
                            unique_hosts_found.add(host)

                    if not is_already_done:
                        scores_for_shard.append(score)

        if not is_already_done:
            np.save(output_path, np.array(scores_for_shard, dtype=np.float32))

    print("\n" + "="*50)
    print("FINAL ALIGNMENT STATISTICS")
    print("="*50)
    print(f"Total Documents Processed:        {global_total_docs:,}")
    print(f"Total Documents Assigned a Score: {global_matched_docs:,}")
    print(f"Unique Hosts Found in Docs:       {len(unique_hosts_found):,}")
    print(f"Unique Hosts Expected from Graph: {target_count:,}")

    missing = target_count - len(unique_hosts_found)
    if missing == 0:
        print("SUCCESS: 100% of hosts from your graph were matched back to the corpus!")
    else:
        print(f"Note: {missing:,} hosts from your graph were not found in this pass.")
    print("="*50)

if __name__ == "__main__":
    generate_annotations()
