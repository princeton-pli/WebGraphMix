import networkx as nx
import nx_cugraph
import cugraph
import nx_parallel
import cugraph.dask as dask_cugraph
import cugraph.dask.comms.comms as Comms
import dask
import dask.dataframe as dd
from dask.distributed import wait
import dask_cuda
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cupy
import pickle
import pandas
import json
import time
import sys
import gc
import os
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from paths import centrality_results_dir, graph_data_dir  # noqa: E402

GRAPH_DATA = graph_data_dir()
RESULTS_DIR = centrality_results_dir()
INPUT_GRAPH_FILE = str(GRAPH_DATA / "corpus_host_graph_undirected.pkl")
OUTPUT_KATZ = str(RESULTS_DIR / "host_graph_scores_katz_maxiter1000.json")
EDGE_PARQUET = str(GRAPH_DATA / "host_graph_edges_for_centrality_calculations.parquet")

# Global flag to stop the monitor thread when done
KEEP_MONITORING = True

def monitor_multi_gpu(client, interval=1.0):
    """Monitor VRAM across ALL GPUs in the Dask cluster."""
    print("Multi-GPU Monitor started...")
    while KEEP_MONITORING:
        try:
            # Gather memory info from all dask workers (GPUs)
            def get_mem():
                import cupy
                free, total = cupy.cuda.Device().mem_info
                return (total - free) / (1024**3), total / (1024**3)
            
            results = client.run(get_mem)
            sys.stdout.write("\n--- GPU Usage ---")
            for addr, (used, total) in results.items():
                sys.stdout.write(f"\n Worker {addr}: {used:.2f}GB / {total:.2f}GB")
            sys.stdout.write("\n" + "-"*20 + "\n")
            sys.stdout.flush()
        except Exception as e:
            pass
        time.sleep(interval)

# 1. configure dask backend (based on what the documentation says)
dask.config.set({
    'distributed.comm.timeouts.connect': '600s', # 10 minutes
    'distributed.comm.timeouts.tcp': '600s',
    'distributed.scheduler.worker-ttl': '86400s', # 24 hours - this is what prevents the workers from timing out!
    'distributed.admin.tick.limit': '10s'
})

def main():
    global KEEP_MONITORING

    # 1. Cluster Initialization
    # Using 80% pool and enabling spill (based on what the documentation says)
    cluster = dask_cuda.LocalCUDACluster(
        rmm_pool_size=0.7,
        memory_limit=0,           # disable the 64GB CPU memory limit
        enable_cudf_spill=True
    )
    client = Client(cluster)
    
    # CRITICAL: Initialize communications for Multi-GPU math
    Comms.initialize(p2p=True)
    print(f"Cluster ready with {len(cluster.workers)} GPUs.")


    # 1. Load Graph
    # print(f"Loading graph from {INPUT_GRAPH_FILE}...")
    # sys.setrecursionlimit(100000)
    # with open(INPUT_GRAPH_FILE, 'rb') as f:
    #     G_cpu = pickle.load(f)
    # df = nx.to_pandas_edgelist(G_cpu)
    # df.to_parquet("commoncrawl/host_graph_edges_for_centrality_calculations.parquet")
    # print(f"Graph loaded: {G_cpu.number_of_nodes()} nodes, {G_cpu.number_of_edges()} edges.")
    # del G_cpu, df

    # 3. Start Monitor
    monitor_thread = threading.Thread(target=monitor_multi_gpu, args=(client, 0.5))
    monitor_thread.daemon = True
    monitor_thread.start()

    # --- 4. DISTRIBUTE DATA (THE FAST WAY) ---
    print("Reading Parquet directly into GPU memory...")
    # Each GPU worker reads its own chunk from the disk directly - otherwise cpu runs out of memory
    ddf = dask_cudf.read_parquet(EDGE_PARQUET)
    wait(ddf) # ensure the load is 100% done on both GPUs

    # --- 5. CREATE MG GRAPH ---
    G_mg = cugraph.Graph(directed=False)
    G_mg.from_dask_cudf_edgelist(ddf, source='source', destination='target')

    # Cleanup to ensure maximum VRAM for the algorithm
    client.cancel(ddf) # Tell Dask to release the memory on workers
    del ddf
    gc.collect()
    client.run(gc.collect) # Force garbage collection on all GPU workers
    
    # =========================================================
    # CONFIGURE PARALLEL BACKEND
    # =========================================================
    # print(f"\nConfiguring Parallel Backend...")

    # print(nx.config)
    # nx.config.backends.parallel.active = True
    # nxp_config = nx.config.backends.parallel
    # nxp_config.n_jobs = 32
    # nxp_config.verbose = 5

    # =========================================================
    # ALGORITHM 1: Standard Katz Centrality (Approximation)
    # =========================================================
    print("\n--- Running Standard Katz Centrality ---")
    print("Parameters: max_iter=1000, tol=1.0e-5")
    
    start = time.time()
    
    # Verbose parameter controls output/progress info from the parallel jobs
    # Works on both Directed and Undirected
    result_ddf = dask_cugraph.katz_centrality(
        G_mg, 
        max_iter=1000, # can adjust to speed up
        tol=1.0e-5
    )
    print(f"Katz centrality calculated in {time.time() - start:.2f} seconds.")

    # 5. COMPUTE AND FORMAT
    # Documentation warns: result_ddf is a dask_cudf.DataFrame.
    # We compute() to bring it to the CPU once the math is done.
    print("Calculation finished. Computing final scores...")
    final_df = result_ddf.compute()
    # Format to dictionary to save as JSON
    katz_scores = final_df.set_index('vertex')['katz_centrality'].to_dict()
    
    print(f"Saving Katz centrality scores to {OUTPUT_KATZ}...")
    with open(OUTPUT_KATZ, 'w') as f:
        json.dump(katz_scores, f)

    KEEP_MONITORING = False
    Comms.destroy()
    client.close()
    cluster.close()

    # # =========================================================
    # # ALGORITHM 2: Approximate Current Flow Betweenness
    # # =========================================================
    # print("\n--- Running Approximate Current Flow Betweenness ---")
    
    # # Requirements: Undirected (Already done) AND Fully Connected
    
    # print("Step A: Ensuring Connectivity (Extracting LCC)...")
    # if nx.is_connected(G):
    #     G_lcc = G
    #     print("Graph is already fully connected.")
    # else:
    #     # Extract Largest Connected Component
    #     lcc_nodes = max(nx.connected_components(G), key=len)
    #     G_lcc = G.subgraph(lcc_nodes).copy()
    #     print(f"Graph was disconnected. Using LCC: {G_lcc.number_of_nodes()} nodes.")
    
    # print("Step B: Running Algorithm...")
    # # Parameters: 
    # # epsilon=0.1 (Fast approximation). Lower is better but much slower.
    # # kmax=10000 (Max samples).
    
    # start = time.time()
    # current_flow_scores = nx.approximate_current_flow_betweenness_centrality(
    #     G_lcc,
    #     normalized=True,
    #     epsilon=0.1, 
    #     kmax=10000,
    #     seed=42
    # )
    # print(f"Current Flow calculated in {time.time() - start:.2f} seconds.")

    # print(f"Saving Current Flow scores to {OUTPUT_CURRENT_FLOW}...")
    # with open(OUTPUT_CURRENT_FLOW, 'w') as f:
    #     json.dump(current_flow_scores, f)

    print("\nAll Done!")

if __name__ == "__main__":
    main()