import argparse
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigvalsh
import random
import csv
import json
import os
import sys


class TransitivityACO:
    def __init__(self, num_nodes, target_edges, target_diam, target_trans):
        self.n = num_nodes
        self.m = target_edges
        self.target_diam = target_diam
        self.target_trans = target_trans
        

        for _ in range(1000):
            self.num_layers = self.target_diam + 1
            valid_cuts = range(1, self.n)
            
            if len(valid_cuts) < self.num_layers - 1:
                self.layer_sizes = [1] * self.n
            else:
                cuts = sorted(random.sample(valid_cuts, self.num_layers - 1))
                cuts = [0] + cuts + [self.n]
                self.layer_sizes = [cuts[i+1] - cuts[i] for i in range(self.num_layers)]
            
            node_indices = np.random.permutation(self.n)
            self.node_to_layer = np.zeros(self.n, dtype=int)
            curr = 0
            for i, size in enumerate(self.layer_sizes):
                self.node_to_layer[node_indices[curr : curr + size]] = i
                curr += size

            u_idx, v_idx = np.triu_indices(self.n, k=1)
            l_u, l_v = self.node_to_layer[u_idx], self.node_to_layer[v_idx]
            valid_mask = np.abs(l_u - l_v) <= 1
            
            self.candidates_u = u_idx[valid_mask]
            self.candidates_v = v_idx[valid_mask]
            self.num_candidates = len(self.candidates_u)
            
            if self.num_candidates >= self.m:
                break
        else:

             raise ValueError("Could not find valid layering configuration.")

        self.is_intra = (self.node_to_layer[self.candidates_u] == self.node_to_layer[self.candidates_v])
        self.is_inter = ~self.is_intra
        self.pheromones = np.ones(self.num_candidates)

    def _compute_metrics(self, adj):
        # 1. Diameter Estimation (Double Sweep)
        csr = scipy.sparse.csr_matrix(adj)
        dists = shortest_path(csr, indices=0, directed=False, unweighted=True)
        if np.isinf(dists).any(): return False, 0, 0
        u = np.argmax(dists)
        dists_u = shortest_path(csr, indices=u, directed=False, unweighted=True)
        d_est = np.max(dists_u)
        valid_d = (d_est <= self.target_diam + 1)
        

        deg = np.sum(adj, axis=1)
        denom = np.sum(deg * (deg - 1))
        
        if denom == 0:
            trans = 0.0
        else:
            numer = np.sum(adj * (adj @ adj))
            trans = numer / denom
        
        return valid_d, d_est, trans

    def _get_laplacian_eigenvalues(self, adj):
        deg = np.sum(adj, axis=1)
        L = np.diag(deg) - adj
        evals = eigvalsh(L)
        return np.sort(evals)

    def run(self, ants=40, max_iter=50):

        found_valid_graphs = [] 
        
        for t in range(max_iter):
            probs = self.pheromones / self.pheromones.sum()
            ant_results = []
            
            for _ in range(ants):
                picks = np.random.choice(self.num_candidates, size=self.m, replace=False, p=probs)
                
                adj = np.zeros((self.n, self.n))
                adj[self.candidates_u[picks], self.candidates_v[picks]] = 1
                adj[self.candidates_v[picks], self.candidates_u[picks]] = 1
                
                valid_d, d_est, trans = self._compute_metrics(adj)
                

                if valid_d and (self.target_trans * 0.95 <= trans <= self.target_trans * 1.05):
                    

                    evals = self._get_laplacian_eigenvalues(adj)
                    edge_list = list(zip(self.candidates_u[picks].tolist(), self.candidates_v[picks].tolist()))
                    

                    found_valid_graphs.append({
                        "edges": edge_list, 
                        "evals": evals,
                        "trans": trans,
                        "diam": d_est
                    })

                ant_results.append((trans, picks, valid_d))


            self.pheromones *= 0.90 
            ant_results.sort(key=lambda x: (not x[2], abs(x[0] - self.target_trans)))
            
            for trans, indices, valid in ant_results[:10]: 
                gap = trans - self.target_trans 
                dist = abs(gap)
                reward = 1.0 / (0.01 + dist)
                if not valid: reward *= 0.1
                
                path_intra = self.is_intra[indices]
                path_inter = self.is_inter[indices]
                
                if gap < 0: 
                    np.add.at(self.pheromones, indices[path_intra], reward * 2.0)
                    np.add.at(self.pheromones, indices[path_inter], reward * 0.5)
                elif gap > 0:
                    np.add.at(self.pheromones, indices[path_inter], reward * 2.0)
                    np.add.at(self.pheromones, indices[path_intra], reward * 0.5)
                else:
                    np.add.at(self.pheromones, indices, reward)


        return found_valid_graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--diam", type=int, required=True)
    parser.add_argument("--cc", type=float, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    run_id = f"D{args.diam}_C{args.cc:.2f}"
    csv_file = os.path.join(args.out_dir, "metrics", f"metrics_{run_id}.csv")
    graph_file = os.path.join(args.out_dir, "graphs", f"graphs_{run_id}.json")

    wins = 0
    total_gap = 0
    valid_graphs_data = []

    for _ in range(args.trials):
        try:
            aco = TransitivityACO(args.N, args.M, args.diam, args.cc)
            graphs = aco.run(ants=40, max_iter=50)
            
            if graphs:
                wins += 1

                best_g = min(graphs, key=lambda x: abs(x['trans'] - args.cc))
                total_gap += (best_g['trans'] - args.cc)
                

                valid_graphs_data.append(best_g)
                
        except Exception as e:

            continue


    success_rate = wins / args.trials if args.trials > 0 else 0
    avg_gap = total_gap / wins if wins > 0 else 0.0


    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow([args.diam, args.cc, success_rate, avg_gap])

    if valid_graphs_data:
        output_data = {
            "target_diam": args.diam,
            "target_cc": args.cc,
            "N": args.N,
            "graphs": []
        }
        for g in valid_graphs_data:
            output_data["graphs"].append({
                "edges": g["edges"],
                "realized_cc": g["trans"],
                "realized_diam": g["diam"],
                "eigenvalues": g["evals"].tolist()
            })
            
        with open(graph_file, "w") as f:
            json.dump(output_data, f)
