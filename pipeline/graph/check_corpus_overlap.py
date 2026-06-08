import zstandard as zstd
import json
import pickle
import io
import glob
import os
import sys
from urllib.parse import urlparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from paths import data_root, graph_data_dir  # noqa: E402

HOST_GRAPH_FILE = graph_data_dir() / "host_url_set.pkl"
DOCS_FOLDER = data_root() / "documents"
# ---------------------

def check_overlap():
    # STEP 1: Load the Host Graph
    print(f"Loading Host Graph from {HOST_GRAPH_FILE}...")
    try:
        with open(HOST_GRAPH_FILE, 'rb') as f:
            valid_hosts = pickle.load(f)
    except FileNotFoundError:
        print("Error: Could not find pickle file. Did you run build_lookup.py?")
        return
    print(f"Graph Loaded! Contains {len(valid_hosts):,} unique hosts.")

    # STEP 2: Find all shard files
    # Looks for any file ending in .jsonl.zst inside the folder
    search_path = os.path.join(DOCS_FOLDER, "*.jsonl.zst")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"Error: No .jsonl.zst files found in {DOCS_FOLDER}")
        return

    print(f"Found {len(files)} shards to process.")

    # Initialize global counters
    total_docs = 0
    matches = 0
    misses = 0
    
    # Setup decompressor
    dctx = zstd.ZstdDecompressor()

    # STEP 3: Loop through every file
    for file_index, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"\n--- Processing File {file_index + 1}/{len(files)}: {filename} ---")
        
        try:
            with open(file_path, 'rb') as fh:
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    
                    for line in text_stream:
                        total_docs += 1
                        
                        try:
                            doc = json.loads(line)
                            
                            # A. Extract URL
                            url = doc.get('url')
                            if not url and 'metadata' in doc:
                                url = doc['metadata'].get('WARC-Target-URI')
                                
                            if not url:
                                misses += 1
                                continue

                            # B. Parse Hostname
                            # "http://news.bbc.co.uk/article" -> "news.bbc.co.uk"
                            parsed = urlparse(url)
                            hostname = parsed.netloc

                            # Clean up port numbers
                            if ':' in hostname:
                                hostname = hostname.split(':')[0]
                            
                            # Clean up "www."
                            hostname_no_www = hostname
                            if hostname.startswith('www.'):
                                hostname_no_www = hostname[4:]

                            # C. Check Overlap
                            if (hostname in valid_hosts) or (hostname_no_www in valid_hosts):
                                matches += 1
                            else:
                                misses += 1

                        except Exception:
                            misses += 1

                        # Progress Report every 100,000 documents (Global Count)
                        if total_docs % 100000 == 0:
                            print(f"  [Global Stats] Scanned: {total_docs:,} | Matches: {matches:,} | "
                                  f"Overlap: {matches/total_docs:.2%}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    # STEP 4: Final Results
    print("\n" + "="*40)
    print("FINAL CUMULATIVE RESULTS")
    print("="*40)
    print(f"Total Files Processed: {len(files)}")
    print(f"Total Documents:       {total_docs:,}")
    print(f"Found in 2025 Graph:   {matches:,}")
    print(f"Missing (Link Rot):    {misses:,}")
    if total_docs > 0:
        print(f"FINAL OVERLAP:         {matches/total_docs:.2%}")
    else:
        print("FINAL OVERLAP:         N/A (No docs found)")

if __name__ == "__main__":
    check_overlap()