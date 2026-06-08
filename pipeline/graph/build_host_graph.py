import gzip
import json
import networkx as nx
import glob
import os
import zstandard as zstd
import io
import pickle
import sys
from urllib.parse import urlparse

# --- LOGGING SETUP ---
class Logger(object):
    def __init__(self, filename="build_graph.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8', buffering=1) # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from paths import data_root, graph_data_dir, repo_root  # noqa: E402

CC_DATA = graph_data_dir()
DOCS_FOLDER = data_root() / "documents"
CC_VERTICES_DIR = CC_DATA / "CC_Graph_Data/vertices"
CC_EDGES_DIR = CC_DATA / "CC_Graph_Data/edges"
OUTPUT_GRAPH_FILE = CC_DATA / "corpus_host_graph_undirected.pkl"

def extract_hostname(url):
    """
    Exact logic from your previous success script:
    1. Parse netloc
    2. Remove port
    3. Remove 'www.'
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc
        
        if not hostname:
            return None

        # Clean up port numbers
        if ':' in hostname:
            hostname = hostname.split(':')[0]
        
        # Clean up "www."
        if hostname.startswith('www.'):
            return hostname[4:]
            
        return hostname
    except:
        return None

def main():
    # --- REDIRECT PRINT STATEMENTS TO FILE ---
    sys.stdout = Logger("build_graph.log")
    print("Logging started. Output is being saved to build_graph.log")
    # -----------------------------------------

    # ---------------------------------------------------------
    # PART A: Scan Corpus-200B for active HOSTS
    # ---------------------------------------------------------
    print("Scanning Corpus-200B for active hostnames...")
    my_hosts = set()
    dctx = zstd.ZstdDecompressor()
    
    files = sorted(glob.glob(os.path.join(DOCS_FOLDER, "*.jsonl.zst")))
    
    for i, file_path in enumerate(files):
        if i % 50 == 0: print(f"Scanned {i}/{len(files)} shards...")
        try:
            with open(file_path, 'rb') as fh:
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    for line in text_stream:
                        try:
                            doc = json.loads(line)
                            
                            url = doc.get('url')
                            if not url and 'metadata' in doc:
                                url = doc['metadata'].get('WARC-Target-URI')
                            
                            if url:
                                hostname = extract_hostname(url)
                                if hostname:
                                    my_hosts.add(hostname)
                        except:
                            continue
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
    
    print(f"Total Unique Hostnames in Corpus-200B: {len(my_hosts):,}")

    # ---------------------------------------------------------
    # PART B: Map CC Graph IDs to Your Hostnames
    # ---------------------------------------------------------
    print("\nMapping Common Crawl Graph IDs...")
    valid_ids = {} # CC_ID -> Hostname string

    vertex_files = sorted(glob.glob(os.path.join(CC_VERTICES_DIR, "*.gz")))
    if not vertex_files:
        print(f"Error: No files in {CC_VERTICES_DIR}. Run script 0 first.")
        return

    for vf in vertex_files:
        print(f"Reading vertices from {os.path.basename(vf)}...")
        with gzip.open(vf, 'rt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                
                cc_id = parts[0]
                rev_host = parts[1] # e.g. "uk.co.bbc.news"
                
                # Reverse it back: "uk.co.bbc.news" -> "news.bbc.co.uk"
                host = ".".join(rev_host.split('.')[::-1])
                
                # Normalize graph node (remove www to match our parsing)
                if host.startswith("www."):
                    host_clean = host[4:]
                else:
                    host_clean = host
                
                # Check Overlap
                if host_clean in my_hosts:
                    valid_ids[cc_id] = host_clean

    # ---------------------------------------------------------
    # STATISTICS: OVERLAP CHECK
    # ---------------------------------------------------------
    matched_count = len(set(valid_ids.values()))
    total_corpus_hosts = len(my_hosts)
    
    print("\n" + "="*40)
    print("OVERLAP STATISTICS")
    print("="*40)
    print(f"Corpus-200B Unique Hosts: {total_corpus_hosts:,}")
    print(f"Hosts found in CC Graph:  {matched_count:,}")
    if total_corpus_hosts > 0:
        overlap_pct = (matched_count / total_corpus_hosts) * 100
        print(f"Coverage:                 {overlap_pct:.2f}%")
    else:
        print("Coverage:                 0%")
    print("="*40 + "\n")

    # ---------------------------------------------------------
    # PART C: Build Graph (UNDIRECTED)
    # ---------------------------------------------------------
    print("Building UNDIRECTED NetworkX Graph...")
    G = nx.Graph() # UNDIRECTED
    
    # Pre-add nodes to ensure we include isolated hosts
    G.add_nodes_from(valid_ids.values()) 

    edge_files = sorted(glob.glob(os.path.join(CC_EDGES_DIR, "*.gz")))
    edge_count = 0
    
    for ef in edge_files:
        print(f"Reading edges from {os.path.basename(ef)}...")
        with gzip.open(ef, 'rt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                
                u_id, v_id = parts[0], parts[1]
                
                # Add edge only if BOTH source and target are in our list
                if u_id in valid_ids and v_id in valid_ids:
                    G.add_edge(valid_ids[u_id], valid_ids[v_id])
                    edge_count += 1

    print(f"\nGraph Construction Complete.")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")

    # ---------------------------------------------------------
    # PART D: Save Graph
    # ---------------------------------------------------------
    print(f"Saving graph object to {OUTPUT_GRAPH_FILE}...")
    sys.setrecursionlimit(100000)
    with open(OUTPUT_GRAPH_FILE, 'wb') as f:
        pickle.dump(G, f)
    print("Done!")

if __name__ == "__main__":
    main()