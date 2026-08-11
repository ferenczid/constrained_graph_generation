#!/usr/bin/env python3
"""
Uniform random layered seed generator.

This is the no-pheromone control for aco_seed_gen.py. It uses the same
layered admissible-edge construction, candidate budget, fast screening,
and exact validation, but every candidate graph is sampled uniformly
from the admissible layered edge set. No pheromone values are stored or
updated.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import scipy.sparse
from scipy.linalg import eigvalsh
from scipy.sparse.csgraph import shortest_path

from metrics_utils import exact_validate_graph


class RandomLayeredConstructor:
    def __init__(
        self,
        num_nodes,
        target_edges,
        target_diam,
        target_trans,
        diam_min=None,
        diam_max=None,
        cc_tol=0.025,
        rng=None,
        max_layer_attempts=1000,
    ):
        self.n = num_nodes
        self.m = target_edges
        self.target_diam = target_diam
        self.target_trans = target_trans
        self.diam_min = target_diam if diam_min is None else diam_min
        self.diam_max = target_diam if diam_max is None else diam_max
        self.cc_tol = cc_tol
        self.rng = np.random.default_rng() if rng is None else rng

        self.num_layers = self.target_diam + 1
        if self.num_layers > self.n:
            raise ValueError(
                f"Cannot create {self.num_layers} nonempty layers with only {self.n} nodes."
            )

        for _ in range(max_layer_attempts):
            if self.num_layers == 1:
                cuts = np.array([], dtype=int)
            else:
                cuts = np.sort(
                    self.rng.choice(
                        np.arange(1, self.n),
                        size=self.num_layers - 1,
                        replace=False,
                    )
                )

            boundaries = np.concatenate(([0], cuts, [self.n]))
            self.layer_sizes = np.diff(boundaries)

            node_indices = self.rng.permutation(self.n)
            self.node_to_layer = np.zeros(self.n, dtype=int)

            current = 0
            for layer, size in enumerate(self.layer_sizes):
                self.node_to_layer[node_indices[current : current + size]] = layer
                current += size

            u_idx, v_idx = np.triu_indices(self.n, k=1)
            layer_u = self.node_to_layer[u_idx]
            layer_v = self.node_to_layer[v_idx]
            valid_mask = np.abs(layer_u - layer_v) <= 1

            self.candidates_u = u_idx[valid_mask]
            self.candidates_v = v_idx[valid_mask]
            self.num_candidates = len(self.candidates_u)

            if self.num_candidates >= self.m:
                break
        else:
            raise ValueError(
                "Could not find a layered configuration with enough admissible edges."
            )

    def _compute_metrics(self, adj):
        """Return the ACO-compatible fast diameter screen and transitivity."""
        csr = scipy.sparse.csr_matrix(adj)

        distances = shortest_path(
            csr,
            indices=0,
            directed=False,
            unweighted=True,
        )
        if np.isinf(distances).any():
            return False, np.inf, 0.0

        farthest = int(np.argmax(distances))
        distances_from_farthest = shortest_path(
            csr,
            indices=farthest,
            directed=False,
            unweighted=True,
        )
        diameter_estimate = float(np.max(distances_from_farthest))
        valid_diameter_screen = diameter_estimate <= self.diam_max + 1

        degrees = np.sum(adj, axis=1)
        denominator = np.sum(degrees * (degrees - 1))

        if denominator == 0:
            transitivity = 0.0
        else:
            numerator = np.sum(adj * (adj @ adj))
            transitivity = float(numerator / denominator)

        return valid_diameter_screen, diameter_estimate, transitivity

    @staticmethod
    def _laplacian_eigenvalues(adj):
        degrees = np.sum(adj, axis=1)
        laplacian = np.diag(degrees) - adj
        return np.sort(eigvalsh(laplacian))

    def run(self, ants=40, max_iter=50):
        """
        Generate ants * max_iter independent random layered candidates.

        The loop structure deliberately matches the ACO candidate budget.
        There are no pheromone updates between iterations.
        """
        exact_checked = 0
        exact_valid = 0
        candidates_generated = 0
        best_graph = None
        best_gap = np.inf

        for _ in range(max_iter):
            for _ in range(ants):
                candidates_generated += 1

                # Uniform sample of m admissible edges without replacement.
                picks = self.rng.choice(
                    self.num_candidates,
                    size=self.m,
                    replace=False,
                )

                adj = np.zeros((self.n, self.n), dtype=float)
                adj[self.candidates_u[picks], self.candidates_v[picks]] = 1.0
                adj[self.candidates_v[picks], self.candidates_u[picks]] = 1.0

                valid_diameter_screen, diameter_estimate, transitivity = (
                    self._compute_metrics(adj)
                )

                # This is only a cheap pre-screen. It is deliberately wider
                # than the exact acceptance interval, so it cannot reject an
                # exactly valid graph.
                cc_screen = abs(transitivity - self.target_trans) <= max(
                    2 * self.cc_tol,
                    0.05,
                )

                if not (valid_diameter_screen and cc_screen):
                    continue

                exact_checked += 1
                verification = exact_validate_graph(
                    adj=adj,
                    target_edges=self.m,
                    diam_min=self.diam_min,
                    diam_max=self.diam_max,
                    target_cc=self.target_trans,
                    cc_tol=self.cc_tol,
                )

                if not verification["valid"]:
                    continue

                exact_valid += 1
                gap = abs(verification["transitivity"] - self.target_trans)

                if gap < best_gap:
                    best_gap = gap
                    edge_list = list(
                        zip(
                            self.candidates_u[picks].tolist(),
                            self.candidates_v[picks].tolist(),
                        )
                    )

                    best_graph = {
                        "edges": edge_list,
                        "evals": self._laplacian_eigenvalues(adj),
                        "trans": verification["transitivity"],
                        "diam_exact": verification["diameter"],
                        "diam_est": diameter_estimate,
                        "exact_valid": True,
                        "verification": verification,
                    }

        return best_graph, exact_checked, exact_valid, candidates_generated


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an exactly validated graph using uniform random layered "
            "construction without pheromone updates."
        )
    )
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--diam", type=int, required=True)
    parser.add_argument("--cc", type=float, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--diam_min", type=int, default=None)
    parser.add_argument("--diam_max", type=int, default=None)
    parser.add_argument("--cc_tol", type=float, default=0.025)
    parser.add_argument("--ants", type=int, default=40)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional base seed. job_id is added to it for independent jobs.",
    )
    args = parser.parse_args()

    if args.N <= 0:
        parser.error("--N must be positive.")
    if args.M < 0 or args.M > args.N * (args.N - 1) // 2:
        parser.error("--M is outside the valid range for a simple graph.")
    if args.trials <= 0 or args.ants <= 0 or args.max_iter <= 0:
        parser.error("--trials, --ants, and --max_iter must be positive.")

    diam_min = args.diam if args.diam_min is None else args.diam_min
    diam_max = args.diam if args.diam_max is None else args.diam_max

    if diam_min > diam_max:
        parser.error("--diam_min cannot exceed --diam_max.")

    seed_value = None
    if args.rng_seed is not None:
        seed_value = args.rng_seed + args.job_id
    rng = np.random.default_rng(seed_value)

    density = args.M / (args.N * (args.N - 1) / 2)
    run_id = (
        f"N{args.N}_M{args.M}_p{density:.3f}_"
        f"D{args.diam}_C{args.cc:.2f}_job{args.job_id}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_dir = os.path.join(args.out_dir, "metrics")
    graphs_dir = os.path.join(args.out_dir, "graphs")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    csv_file = os.path.join(metrics_dir, f"metrics_random_layered_{run_id}.csv")
    graph_file = os.path.join(graphs_dir, f"graphs_random_layered_{run_id}.json")
    seed_file = os.path.join(args.out_dir, f"seed_{args.job_id}.json")

    wins = 0
    total_gap = 0.0
    exact_checked = 0
    exact_valid = 0
    candidates_generated = 0
    valid_graphs = []

    start_time = time.time()

    for _ in range(args.trials):
        try:
            constructor = RandomLayeredConstructor(
                num_nodes=args.N,
                target_edges=args.M,
                target_diam=args.diam,
                target_trans=args.cc,
                diam_min=diam_min,
                diam_max=diam_max,
                cc_tol=args.cc_tol,
                rng=rng,
            )

            best_graph, checked, valid, generated = constructor.run(
                ants=args.ants,
                max_iter=args.max_iter,
            )

            exact_checked += checked
            exact_valid += valid
            candidates_generated += generated

            if best_graph is not None:
                wins += 1
                total_gap += best_graph["trans"] - args.cc
                valid_graphs.append(best_graph)

        except Exception as exc:
            print(f"Trial failed: {exc}", file=sys.stderr)

    runtime_total = time.time() - start_time
    success_rate = wins / args.trials
    average_gap = total_gap / wins if wins else 0.0
    runtime_per_trial = runtime_total / args.trials
    exact_valid_fraction = exact_valid / exact_checked if exact_checked else 0.0
    mean_exact_diameter = (
        float(np.mean([graph["diam_exact"] for graph in valid_graphs]))
        if valid_graphs
        else np.nan
    )
    mean_exact_cc = (
        float(np.mean([graph["trans"] for graph in valid_graphs]))
        if valid_graphs
        else np.nan
    )

    with open(csv_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "generator",
                "N",
                "M",
                "density",
                "target_diam",
                "diam_min",
                "diam_max",
                "target_cc",
                "cc_tol",
                "trials",
                "wins",
                "success_rate",
                "average_gap",
                "runtime_total",
                "runtime_per_trial",
                "candidates_generated",
                "exact_checked",
                "exact_valid",
                "exact_valid_fraction",
                "mean_exact_diameter",
                "mean_exact_cc",
                "ants",
                "max_iter",
                "job_id",
                "rng_seed",
            ]
        )
        writer.writerow(
            [
                "random_layered",
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
                average_gap,
                runtime_total,
                runtime_per_trial,
                candidates_generated,
                exact_checked,
                exact_valid,
                exact_valid_fraction,
                mean_exact_diameter,
                mean_exact_cc,
                args.ants,
                args.max_iter,
                args.job_id,
                seed_value,
            ]
        )

    if not valid_graphs:
        print(
            "No exactly valid random-layered seed found for "
            f"N={args.N}, M={args.M}, D=[{diam_min},{diam_max}], "
            f"C={args.cc}±{args.cc_tol} after "
            f"{candidates_generated} candidates.",
            file=sys.stderr,
        )
        sys.exit(2)

    seed_graph = min(
        valid_graphs,
        key=lambda graph: abs(graph["trans"] - args.cc),
    )

    output_data = {
        "generator": "random_layered",
        "pheromone_updates": False,
        "target_diam": args.diam,
        "diam_min": diam_min,
        "diam_max": diam_max,
        "target_cc": args.cc,
        "cc_tol": args.cc_tol,
        "N": args.N,
        "M": args.M,
        "density": density,
        "ants": args.ants,
        "max_iter": args.max_iter,
        "candidate_budget_per_trial": args.ants * args.max_iter,
        "graphs": [
            {
                "edges": graph["edges"],
                "realized_cc": graph["trans"],
                "realized_diam": graph["diam_exact"],
                "diam_est": graph["diam_est"],
                "exact_valid": graph["exact_valid"],
                "verification": graph["verification"],
                "eigenvalues": graph["evals"].tolist(),
            }
            for graph in valid_graphs
        ],
    }

    with open(graph_file, "w") as handle:
        json.dump(output_data, handle, indent=2)

    # B/H/rho are retained as neutral numeric metadata for compatibility
    # with code that may inspect the ACO seed schema. They are not used.
    seed_output = {
        "generator": "random_layered",
        "pheromone_updates": False,
        "N": args.N,
        "M": args.M,
        "density": density,
        "target_diam": args.diam,
        "diam_min": diam_min,
        "diam_max": diam_max,
        "target_cc": args.cc,
        "cc_tol": args.cc_tol,
        "B": 1.0,
        "H": 1.0,
        "rho": 1.0,
        "ants": args.ants,
        "max_iter": args.max_iter,
        "candidate_budget_per_trial": args.ants * args.max_iter,
        "job_id": args.job_id,
        "rng_seed": seed_value,
        "edges": seed_graph["edges"],
        "realized_cc": seed_graph["trans"],
        "realized_diam": seed_graph["diam_exact"],
        "diam_est": seed_graph["diam_est"],
        "exact_valid": True,
        "verification": seed_graph["verification"],
    }

    with open(seed_file, "w") as handle:
        json.dump(seed_output, handle, indent=2)

    print(f"Saved random-layered MCMC seed to {seed_file}")


if __name__ == "__main__":
    main()
