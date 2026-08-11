import argparse
import json
import numpy as np
import networkx as nx
import sys
import os


try:
    import clust_diam_MCMC as mcmc
except ImportError:
    print("Error: clust_diam_MCMC.py not found in directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_seed", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--swaps", type=int, default=10000)
    args = parser.parse_args()

    # 1. Load ACO Seed
    with open(args.input_seed, 'r') as f:
        data = json.load(f)
    
    N = data['N']
    edges = data['edges']
    
    # Build Matrix for MCMC Library
    mtx = np.zeros((N, N), dtype=int)
    for u, v in edges:
        mtx[u, v] = 1
        mtx[v, u] = 1
        
    # 2. Run MCMC Shuffling
    # Using the library function you provided
    shuffled_mtx = mcmc.shuffle_mtx(mtx, swaps=args.swaps, tolerance=0.10)
    
    # 3. Save Result
    # Convert back to Edge List for consistent JSON storage
    G_final = nx.from_numpy_array(shuffled_mtx)
    final_edges = list(G_final.edges())
    
    output_data = {
        "N": N,
        "edges": final_edges,
        "metrics": mcmc.graph_similarity(mtx, shuffled_mtx)
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()
