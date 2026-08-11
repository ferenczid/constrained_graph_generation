import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigvalsh

from metrics_utils import exact_validate_graph


class TransitivityACO:
    def __init__(
        self,
        num_nodes,
        target_edges,
        target_diam,
        target_trans,
        diam_min=None,
        diam_max=None,
        cc_tol=0.025,
        rho=0.90,
        B=2.0,
        H=0.5,
        save_eigs=False,
    ):
        self.n = num_nodes
        self.m = target_edges
        self.target_diam = target_diam
        self.target_trans = target_trans
        self.rho = rho
        self.B = B
        self.H = H
        self.save_eigs = save_eigs

        # Exact-validation bounds.
        self.diam_min = target_diam if diam_min is None else diam_min
        self.diam_max = target_diam if diam_max is None else diam_max
        self.cc_tol = cc_tol

        # Random layered candidate set: only intra-layer and adjacent-layer edges.
        for _ in range(1000):
            self.num_layers = self.target_diam + 1
            valid_cuts = range(1, self.n)

            if len(valid_cuts) < self.num_layers - 1:
                self.layer_sizes = [1] * self.n
            else:
                cuts = sorted(random.sample(valid_cuts, self.num_layers - 1))
                cuts = [0] + cuts + [self.n]
                self.layer_sizes = [cuts[i + 1] - cuts[i] for i in range(self.num_layers)]

            node_indices = np.random.permutation(self.n)
            self.node_to_layer = np.zeros(self.n, dtype=int)
            curr = 0
            for i, size in enumerate(self.layer_sizes):
                self.node_to_layer[node_indices[curr: curr + size]] = i
                curr += size

            u_idx, v_idx = np.triu_indices(self.n, k=1)
            l_u = self.node_to_layer[u_idx]
            l_v = self.node_to_layer[v_idx]
            valid_mask = np.abs(l_u - l_v) <= 1

            self.candidates_u = u_idx[valid_mask]
            self.candidates_v = v_idx[valid_mask]
            self.num_candidates = len(self.candidates_u)

            if self.num_candidates >= self.m:
                break
        else:
            raise ValueError("Could not find valid layering configuration.")

        self.is_intra = (
            self.node_to_layer[self.candidates_u]
            == self.node_to_layer[self.candidates_v]
        )
        self.is_inter = ~self.is_intra
        self.pheromones = np.ones(self.num_candidates, dtype=float)

    def _compute_metrics_fast(self, adj):
        """Fast search-time screen: double-sweep diameter estimate + transitivity."""
        csr = scipy.sparse.csr_matrix(adj)
        dists = shortest_path(csr, indices=0, directed=False, unweighted=True)
        if np.isinf(dists).any():
            return False, 0.0, 0.0

        u = int(np.argmax(dists))
        dists_u = shortest_path(csr, indices=u, directed=False, unweighted=True)
        d_est = float(np.max(dists_u))

        # Deliberately slightly permissive: exact validation happens later.
        valid_d = d_est <= self.diam_max + 1

        deg = np.sum(adj, axis=1)
        denom = np.sum(deg * (deg - 1))
        if denom == 0:
            trans = 0.0
        else:
            numer = np.sum(adj * (adj @ adj))
            trans = float(numer / denom)

        return valid_d, d_est, trans

    def _get_laplacian_eigenvalues(self, adj):
        deg = np.sum(adj, axis=1)
        L = np.diag(deg) - adj
        evals = eigvalsh(L)
        return np.sort(evals)

    def run(self, ants=40, max_iter=50):
        exact_checked = 0
        exact_valid = 0
        found_valid_graphs = []

        for _ in range(max_iter):
            probs = self.pheromones / self.pheromones.sum()
            ant_results = []

            for _ in range(ants):
                picks = np.random.choice(
                    self.num_candidates,
                    size=self.m,
                    replace=False,
                    p=probs,
                )

                adj = np.zeros((self.n, self.n), dtype=np.int8)
                adj[self.candidates_u[picks], self.candidates_v[picks]] = 1
                adj[self.candidates_v[picks], self.candidates_u[picks]] = 1

                valid_d, d_est, trans = self._compute_metrics_fast(adj)

                # Fast screen only; exact validation determines saved/counted graphs.
                cc_screen = abs(trans - self.target_trans) <= max(2 * self.cc_tol, 0.05)

                if valid_d and cc_screen:
                    exact_checked += 1
                    verification = exact_validate_graph(
                        adj=adj,
                        target_edges=self.m,
                        diam_min=self.diam_min,
                        diam_max=self.diam_max,
                        target_cc=self.target_trans,
                        cc_tol=self.cc_tol,
                    )

                    if verification["valid"]:
                        exact_valid += 1
                        edge_list = list(zip(
                            self.candidates_u[picks].tolist(),
                            self.candidates_v[picks].tolist(),
                        ))

                        record = {
                            "edges": edge_list,
                            "trans": verification["transitivity"],
                            "diam_exact": verification["diameter"],
                            "diam_est": d_est,
                            "exact_valid": True,
                            "verification": verification,
                        }
                        if self.save_eigs:
                            record["evals"] = self._get_laplacian_eigenvalues(adj)

                        found_valid_graphs.append(record)

                ant_results.append((trans, picks, valid_d))

            # Pheromone evaporation and elite reinforcement.
            self.pheromones *= self.rho
            ant_results.sort(key=lambda x: (not x[2], abs(x[0] - self.target_trans)))

            for trans, indices, valid in ant_results[:10]:
                gap = trans - self.target_trans
                dist = abs(gap)
                reward = 1.0 / (0.01 + dist)
                if not valid:
                    reward *= 0.1

                path_intra = self.is_intra[indices]
                path_inter = self.is_inter[indices]

                if gap < 0:
                    # Clustering too low: reinforce intra-layer edges, hinder inter-layer edges.
                    np.add.at(self.pheromones, indices[path_intra], reward * self.B)
                    np.add.at(self.pheromones, indices[path_inter], reward * self.H)
                elif gap > 0:
                    # Clustering too high: reinforce inter-layer edges, hinder intra-layer edges.
                    np.add.at(self.pheromones, indices[path_inter], reward * self.B)
                    np.add.at(self.pheromones, indices[path_intra], reward * self.H)
                else:
                    np.add.at(self.pheromones, indices, reward)

        return found_valid_graphs, exact_checked, exact_valid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--diam", type=int, required=True)
    parser.add_argument("--cc", type=float, required=True)
    parser.add_argument("--out_dir", type=str, default="BH_sensitivity_results")
    parser.add_argument("--diam_min", type=int, default=None)
    parser.add_argument("--diam_max", type=int, default=None)
    parser.add_argument("--cc_tol", type=float, default=0.025)
    parser.add_argument("--ants", type=int, default=40)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--B", type=float, default=2.0)
    parser.add_argument("--H", type=float, default=0.5)
    parser.add_argument("--setting_label", type=str, default="")
    parser.add_argument("--save_eigs", action="store_true")
    parser.add_argument("--no_save_graphs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    diam_min = args.diam if args.diam_min is None else args.diam_min
    diam_max = args.diam if args.diam_max is None else args.diam_max
    density = args.M / (args.N * (args.N - 1) / 2)

    safe_label = args.setting_label.replace("/", "_").replace(" ", "_")
    label_part = f"_{safe_label}" if safe_label else ""
    run_id = (
        f"N{args.N}_M{args.M}_p{density:.3f}"
        f"_D{args.diam}_C{args.cc:.2f}"
        f"_B{args.B:.2f}_H{args.H:.2f}"
        f"_rho{args.rho:.2f}_A{args.ants}_T{args.max_iter}"
        f"{label_part}"
    )

    metrics_dir = os.path.join(args.out_dir, "metrics")
    graphs_dir = os.path.join(args.out_dir, "graphs")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    csv_file = os.path.join(metrics_dir, f"metrics_{run_id}.csv")
    graph_file = os.path.join(graphs_dir, f"graphs_{run_id}.json")

    wins = 0
    total_gap = 0.0
    valid_graphs_data = []
    exact_checked = 0
    exact_valid = 0
    start_time = time.time()

    for trial in range(args.trials):
        try:
            aco = TransitivityACO(
                args.N,
                args.M,
                args.diam,
                args.cc,
                diam_min=diam_min,
                diam_max=diam_max,
                cc_tol=args.cc_tol,
                rho=args.rho,
                B=args.B,
                H=args.H,
                save_eigs=args.save_eigs,
            )

            graphs, checked, valid = aco.run(
                ants=args.ants,
                max_iter=args.max_iter,
            )

            exact_checked += checked
            exact_valid += valid

            if graphs:
                wins += 1
                best_g = min(graphs, key=lambda x: abs(x["trans"] - args.cc))
                total_gap += best_g["trans"] - args.cc
                valid_graphs_data.append(best_g)

        except Exception as e:
            print(f"Trial {trial} failed: {e}", file=sys.stderr)
            continue

    success_rate = wins / args.trials if args.trials > 0 else 0.0
    avg_gap = total_gap / wins if wins > 0 else 0.0
    runtime_total = time.time() - start_time
    runtime_per_trial = runtime_total / args.trials if args.trials > 0 else 0.0
    exact_valid_fraction = exact_valid / exact_checked if exact_checked > 0 else 0.0

    mean_exact_diam = (
        float(np.mean([g["diam_exact"] for g in valid_graphs_data]))
        if valid_graphs_data else np.nan
    )
    mean_exact_cc = (
        float(np.mean([g["trans"] for g in valid_graphs_data]))
        if valid_graphs_data else np.nan
    )

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            args.N,
            args.M,
            density,
            args.diam,
            diam_min,
            diam_max,
            args.cc,
            args.cc_tol,
            args.trials,
            wins,
            success_rate,
            avg_gap,
            runtime_total,
            runtime_per_trial,
            exact_checked,
            exact_valid,
            exact_valid_fraction,
            mean_exact_diam,
            mean_exact_cc,
            args.B,
            args.H,
            args.rho,
            args.ants,
            args.max_iter,
            args.setting_label,
        ])

    if valid_graphs_data and not args.no_save_graphs:
        output_data = {
            "setting_label": args.setting_label,
            "target_diam": args.diam,
            "diam_min": diam_min,
            "diam_max": diam_max,
            "target_cc": args.cc,
            "cc_tol": args.cc_tol,
            "N": args.N,
            "M": args.M,
            "density": density,
            "B": args.B,
            "H": args.H,
            "rho": args.rho,
            "ants": args.ants,
            "max_iter": args.max_iter,
            "save_eigs": args.save_eigs,
            "graphs": [],
        }

        for g in valid_graphs_data:
            graph_record = {
                "edges": g["edges"],
                "realized_cc": g["trans"],
                "realized_diam": g["diam_exact"],
                "diam_est": g["diam_est"],
                "exact_valid": g["exact_valid"],
                "verification": g["verification"],
            }
            if args.save_eigs and "evals" in g:
                graph_record["eigenvalues"] = g["evals"].tolist()

            output_data["graphs"].append(graph_record)

        with open(graph_file, "w") as f:
            json.dump(output_data, f)


if __name__ == "__main__":
    main()
