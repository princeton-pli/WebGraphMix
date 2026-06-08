import gzip
import os
import sys
import requests
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from paths import graph_data_dir, repo_root  # noqa: E402

# Paper uses cc-main-2023-24-sep-nov-feb-host (see https://commoncrawl.org/web-graphs)
CC_DATA = graph_data_dir()
VERTICES_PATHS_FILE = CC_DATA / "cc_host_graph_versions/cc-main-2023-24-sep-nov-feb-host-vertices.paths.gz"
EDGES_PATHS_FILE = CC_DATA / "cc_host_graph_versions/cc-main-2023-24-sep-nov-feb-host-edges.paths.gz"
VERTICES_DIR = CC_DATA / "CC_Graph_Data/vertices"
EDGES_DIR = CC_DATA / "CC_Graph_Data/edges"

BASE_URL = "https://data.commoncrawl.org/"

def download_files(paths_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading paths from {paths_file}...")
    
    urls_to_download = []
    # Read the text file inside the .gz
    with gzip.open(paths_file, 'rt', encoding='utf-8') as f:
        for line in f:
            relative_path = line.strip()
            if relative_path:
                urls_to_download.append(relative_path)

    print(f"Found {len(urls_to_download)} files. Downloading to {output_dir}...")

    for i, rel_path in enumerate(urls_to_download):
        filename = os.path.basename(rel_path)
        local_path = os.path.join(output_dir, filename)
        full_url = BASE_URL + rel_path
        
        if os.path.exists(local_path):
            print(f"[{i+1}/{len(urls_to_download)}] Skipping {filename} (Exists)")
            continue

        print(f"[{i+1}/{len(urls_to_download)}] Downloading {filename}...")
        
        try:
            with requests.get(full_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

def main():
    os.chdir(repo_root())
    print("--- Step 1: Downloading Vertex Files ---")
    download_files(str(VERTICES_PATHS_FILE), str(VERTICES_DIR))

    print("\n--- Step 2: Downloading Edge Files ---")
    download_files(str(EDGES_PATHS_FILE), str(EDGES_DIR))
    
    print("\nAll downloads complete!")

if __name__ == "__main__":
    main()