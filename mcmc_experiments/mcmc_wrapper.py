import argparse
import json
import numpy as np
import networkx as nx
import sys
import os
import time
from metrics_utils import exact_validate_graph


try:
    import clust_diam_MCMC as mcmc
except ImportError:
    print("Error: clust_diam_MCMC.py not found in directory.")
    sys.exit(1)
    
    
def to_jsonable(v):
    if isinstance(v, dict):
        return {k: to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def build_matrix(N, edges):
    mtx = np.zeros((N, N), dtype=int)

    for u, v in edges:
        u = int(u)
        v = int(v)
        mtx[u, v] = 1
        mtx[v, u] = 1

    return mtx

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_seed", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10000,
        help="Number of raw MCMC proposal iterations, including rejections.",
    )

    # Optional overrides. Normally these come from the seed JSON.
    parser.add_argument("--diam_min", type=int, default=None)
    parser.add_argument("--diam_max", type=int, default=None)
    parser.add_argument("--target_cc", type=float, default=None)
    parser.add_argument("--cc_tol", type=float, default=None)

    args = parser.parse_args()

    if args.n_iter < 0:
        raise ValueError("--n_iter must be non-negative.")

    # 1. Load exact-valid ACO seed
    with open(args.input_seed, "r") as f:
        data = json.load(f)

    N = int(data["N"])
    edges = data["edges"]
    initial_edges = [[int(u), int(v)] for u, v in edges]
    M = int(data.get("M", len(edges)))

    diam_min = args.diam_min if args.diam_min is not None else int(data["diam_min"])
    diam_max = args.diam_max if args.diam_max is not None else int(data["diam_max"])
    target_cc = args.target_cc if args.target_cc is not None else float(data["target_cc"])
    cc_tol = args.cc_tol if args.cc_tol is not None else float(data.get("cc_tol", 0.025))

    mtx = build_matrix(N, edges)

    # 2. Verify seed before MCMC
    seed_verification = exact_validate_graph(
        adj=mtx,
        target_edges=M,
        diam_min=diam_min,
        diam_max=diam_max,
        target_cc=target_cc,
        cc_tol=cc_tol,
    )

    if not seed_verification["valid"]:
        raise ValueError(
            "Input seed is not exact-valid under the requested constraints. "
            f"Seed file: {args.input_seed}. "
            f"Verification: {seed_verification}"
        )

    # 3. Run exact-preserving MCMC
    start_time = time.perf_counter()
    shuffled_mtx, diagnostics = mcmc.shuffle_mtx_exact(
        mtx,
        n_iter=args.n_iter,
        target_edges=M,
        target_cc=target_cc,
        cc_tol=cc_tol,
        diam_min=diam_min,
        diam_max=diam_max,
    )
    runtime_seconds = time.perf_counter() - start_time

    # 4. Verify final graph
    final_verification = exact_validate_graph(
        adj=shuffled_mtx,
        target_edges=M,
        diam_min=diam_min,
        diam_max=diam_max,
        target_cc=target_cc,
        cc_tol=cc_tol,
    )

    if not final_verification["valid"]:
        raise RuntimeError(
            "Final graph failed exact validation in wrapper: "
            f"{final_verification}"
        )

    # 5. Save result
    G_final = nx.from_numpy_array(shuffled_mtx)
    final_edges = [[int(u), int(v)] for u, v in G_final.edges()]

    metrics = mcmc.graph_similarity(mtx, shuffled_mtx)

    output_data = {
        "N": N,
        "M": M,
        "input_seed": args.input_seed,
        "initial_edges": initial_edges,

        "n_iter_requested": args.n_iter,
        "runtime_seconds": runtime_seconds,

        "target_diam": data.get("target_diam", None),
        "diam_min": diam_min,
        "diam_max": diam_max,
        "target_cc": target_cc,
        "cc_tol": cc_tol,

        "seed_valid": bool(seed_verification["valid"]),
        "final_valid": bool(final_verification["valid"]),

        "seed_verification": to_jsonable(seed_verification),
        "final_verification": to_jsonable(final_verification),
        "mcmc_diagnostics": to_jsonable(diagnostics),

        "metrics": to_jsonable(metrics),
        "edges": final_edges,
    }

    out_parent = os.path.dirname(args.output_file)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(
        f"Saved exact MCMC sample to {args.output_file}. "
        f"iterations={diagnostics['proposal_attempts']}, "
        f"accepted={diagnostics['accepted']}, "
        f"acceptance_rate={diagnostics['acceptance_rate']:.4f}, "
        f"runtime={runtime_seconds:.2f}s, "
        f"exact_checked={diagnostics['exact_checked']}, "
        f"final_valid={final_verification['valid']}"
    )

if __name__ == "__main__":
    main()
