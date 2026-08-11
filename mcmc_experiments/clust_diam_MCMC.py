import numpy as np
import random
import scipy.sparse
import scipy.sparse.csgraph
from metrics_utils import exact_validate_graph

def exact_valid_mtx(mtx, target_edges, diam_min, diam_max, target_cc, cc_tol):
    verification = exact_validate_graph(
        adj=mtx,
        target_edges=target_edges,
        diam_min=diam_min,
        diam_max=diam_max,
        target_cc=target_cc,
        cc_tol=cc_tol,
    )
    return verification["valid"], verification

def deg_of(A):
    return np.sum(A, axis=1)

def cherries(A):
    d = deg_of(A)
    return np.sum(d * (d - 1) * 0.5)

def triangles(A):
    A2 = A @ A
    return np.trace(A @ A2) / 2

def cc_mtx(A):
    c = cherries(A)
    if c == 0: return 0.0
    return triangles(A) / c

def edge_list(A):
    rows, cols = np.triu_indices(len(A), k=1)
    is_edge = A[rows, cols] == 1
    edges = np.column_stack((rows[is_edge], cols[is_edge]))
    non_edges = np.column_stack((rows[~is_edge], cols[~is_edge]))
    return edges, non_edges

def _json_safe_verification(v):

    if isinstance(v, dict):
        return {k: _json_safe_verification(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe_verification(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def get_exact_diameter(edges_array, num_nodes):
    # Calculates Exact Diameter (All-Pairs). 
    # Used ONLY for the baseline.
    data = np.ones(edges_array.shape[0], dtype=np.int8)
    graph_sparse = scipy.sparse.coo_matrix(
        (data, (edges_array[:, 0], edges_array[:, 1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Dense matrix return (heavy memory, accurate)
    dists_matrix = scipy.sparse.csgraph.shortest_path(
        graph_sparse, 
        directed=False, 
        unweighted=True
    )
    
    finite_mask = np.isfinite(dists_matrix)
    if not finite_mask.any(): return 0
    return int(dists_matrix[finite_mask].max())

def get_pseudo_diameter(edges_array, num_nodes):

    
    data = np.ones(edges_array.shape[0], dtype=np.int8)
    graph_sparse = scipy.sparse.coo_matrix(
        (data, (edges_array[:, 0], edges_array[:, 1])), 
        shape=(num_nodes, num_nodes)
    )
    

    start_node = random.randrange(num_nodes)
    dists_1 = scipy.sparse.csgraph.shortest_path(
        graph_sparse, directed=False, unweighted=True, indices=start_node
    )
    

    if not np.all(np.isfinite(dists_1)):
        return 0 # Penalize disconnected graphs
        
    farthest_node = np.argmax(dists_1)
    

    dists_2 = scipy.sparse.csgraph.shortest_path(
        graph_sparse, directed=False, unweighted=True, indices=farthest_node
    )
    
    return int(np.nanmax(dists_2))

def pick_node(mtx, edges, non_edges, degrees, triangles, cherries, target_cc, flex=0.01, diam_lower=0, diam_upper=10000):
    N = len(mtx)
    
    if cherries == 0: return 'TRIVIAL GRAPH'
        
    while True:

        idx_e = random.randrange(edges.shape[0])
        x, y = edges[idx_e]
        
        idx_ne = random.randrange(non_edges.shape[0])
        u, v = non_edges[idx_ne]
        

        if u == x or u == y or v == x or v == y:
            continue
            

        triangles_new = triangles + 3 * (np.dot(mtx[u], mtx[v]) - np.dot(mtx[x], mtx[y]))
        cherries_new = cherries + degrees[u] + degrees[v] - (degrees[x] + degrees[y] - 2)
        
        if cherries_new == 0: continue
        cc_new = triangles_new / cherries_new
        
        if np.abs(cc_new - target_cc) > flex:
            continue
            

        original_edge_x = edges[idx_e, 0]
        original_edge_y = edges[idx_e, 1]
        

        edges[idx_e, 0] = u
        edges[idx_e, 1] = v

        approx_diam = get_pseudo_diameter(edges, N)
        

        edges[idx_e, 0] = original_edge_x
        edges[idx_e, 1] = original_edge_y
        

        if approx_diam > diam_upper or approx_diam < diam_lower:
            continue
            
        return x, y, u, v, idx_e, idx_ne, triangles_new, cherries_new


def pick_node_exact(
    mtx,
    edges,
    non_edges,
    degrees,
    tri_num,
    cherry_num,
    target_edges,
    target_cc,
    cc_tol,
    diam_min,
    diam_max,
    max_attempts=10000,
    diagnostics=None,
):
    """
    Exact-preserving version of pick_node.

    Keeps the current single-edge relocation proposal:
        remove one existing edge (x,y)
        add one non-edge (u,v)

    First screens by exact local transitivity update.
    Then temporarily applies the move and calls exact_validate_graph.
    Returns a valid move, or None if no valid move was found within
    max_attempts. The MH sampler calls this with max_attempts=1, so every
    rejected proposal becomes a self-transition.
    """
    N = len(mtx)

    if cherry_num == 0:
        return "TRIVIAL GRAPH"

    for _ in range(max_attempts):
        if diagnostics is not None:
            diagnostics["proposal_attempts"] += 1

        idx_e = random.randrange(edges.shape[0])
        x, y = edges[idx_e]

        idx_ne = random.randrange(non_edges.shape[0])
        u, v = non_edges[idx_ne]

        # Keep the same four-distinct-endpoint restriction as the old code.
        if u == x or u == y or v == x or v == y:
            if diagnostics is not None:
                diagnostics["rejected_endpoint_overlap"] += 1
            continue

        # Local exact update for transitivity numerator/denominator.
        tri_new = tri_num + 3 * (np.dot(mtx[u], mtx[v]) - np.dot(mtx[x], mtx[y]))
        cherry_new = cherry_num + degrees[u] + degrees[v] - (degrees[x] + degrees[y] - 2)

        if cherry_new == 0:
            if diagnostics is not None:
                diagnostics["rejected_zero_cherries"] += 1
            continue

        cc_new = tri_new / cherry_new

        if abs(cc_new - target_cc) > cc_tol:
            if diagnostics is not None:
                diagnostics["rejected_cc"] += 1
            continue

        if diagnostics is not None:
            diagnostics["cc_screen_passed"] += 1

        # Temporarily apply the move.
        mtx[x, y] = 0
        mtx[y, x] = 0
        mtx[u, v] = 1
        mtx[v, u] = 1

        if diagnostics is not None:
            diagnostics["exact_checked"] += 1

        try:
            verification = exact_validate_graph(
                adj=mtx,
                target_edges=target_edges,
                diam_min=diam_min,
                diam_max=diam_max,
                target_cc=target_cc,
                cc_tol=cc_tol,
            )
        finally:
            # Revert the temporary move even if validation raises an error.
            mtx[x, y] = 1
            mtx[y, x] = 1
            mtx[u, v] = 0
            mtx[v, u] = 0

        if verification["valid"]:
            return (
                x,
                y,
                u,
                v,
                idx_e,
                idx_ne,
                tri_new,
                cherry_new,
                _json_safe_verification(verification),
            )

        if diagnostics is not None:
            diagnostics["rejected_exact"] += 1

    return None


def shuffle_mtx(*args, **kwargs):
    """Disabled legacy heuristic sampler.

    The production experiment must use ``shuffle_mtx_exact``, where one
    Markov-chain iteration equals one raw proposal and rejected proposals are
    retained as self-transitions.
    """
    raise RuntimeError(
        "Legacy heuristic sampler is disabled. "
        "Use shuffle_mtx_exact with a raw proposal count."
    )

def graph_similarity(adj1, adj2):
    A = np.array(adj1)
    B = np.array(adj2)
    v1 = A.flatten(); v2 = B.flatten()
    results = {}
    results['Hamming Distance'] = np.mean(np.abs(A - B))
    
    inter = np.sum(np.minimum(A, B))
    union = np.sum(np.maximum(A, B))
    results['Jaccard Similarity'] = inter / union if union != 0 else 1.0

    if np.std(v1) == 0 or np.std(v2) == 0: results['Correlation'] = 0.0
    else: results['Correlation'] = np.corrcoef(v1, v2)[0, 1]

    L1 = np.diag(np.sum(A, axis=1)) - A
    L2 = np.diag(np.sum(B, axis=1)) - B
    eig1 = np.linalg.eigvalsh(L1)
    eig2 = np.linalg.eigvalsh(L2)
    results['Spectral Distance (Laplacian)'] = np.linalg.norm(eig1 - eig2)
    return results
    
    
    
def shuffle_mtx_exact(
    mtx,
    n_iter=100,
    target_edges=None,
    target_cc=None,
    cc_tol=0.025,
    diam_min=None,
    diam_max=None,
    progress_every=1000,
):
    """
    Exact-constrained Metropolis--Hastings sampler.

    Each iteration makes exactly one uniformly sampled edge--non-edge
    relocation proposal. Proposals with overlapping endpoints or proposals
    that violate the exact constraints are rejected as self-transitions.

    n_iter = number of raw proposal attempts, including rejected proposals.
    """
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative.")

    mtx = mtx.copy().astype(int)
    N = len(mtx)

    if target_edges is None:
        target_edges = int(np.sum(mtx) // 2)

    if target_cc is None or diam_min is None or diam_max is None:
        raise ValueError(
            "shuffle_mtx_exact requires target_cc, diam_min, and diam_max. "
            "Do not use seed-derived tolerance for the corrected sampler."
        )

    target_edges = int(target_edges)
    diam_min = int(diam_min)
    diam_max = int(diam_max)
    target_cc = float(target_cc)
    cc_tol = float(cc_tol)

    seed_verification = exact_validate_graph(
        adj=mtx,
        target_edges=target_edges,
        diam_min=diam_min,
        diam_max=diam_max,
        target_cc=target_cc,
        cc_tol=cc_tol,
    )

    if not seed_verification["valid"]:
        raise ValueError(
            "Input seed is not exact-valid under the requested constraints: "
            f"{seed_verification}"
        )

    edges, non_edges = edge_list(mtx)
    degrees = deg_of(mtx)
    tri_num = triangles(mtx)
    cherry_num = cherries(mtx)

    diagnostics = {
        "requested_iterations": int(n_iter),
        "accepted": 0,
        "proposal_attempts": 0,
        "cc_screen_passed": 0,
        "exact_checked": 0,
        "rejected_endpoint_overlap": 0,
        "rejected_zero_cherries": 0,
        "rejected_cc": 0,
        "rejected_exact": 0,
        "stopped_early": False,
        "failed_at_step": None,
        "target_edges": target_edges,
        "target_cc": target_cc,
        "cc_tol": cc_tol,
        "diam_min": diam_min,
        "diam_max": diam_max,
        "seed_verification": _json_safe_verification(seed_verification),
        "last_accepted_verification": _json_safe_verification(seed_verification),
    }

    print(
        f"Exact MCMC target: M={target_edges}, "
        f"D=[{diam_min},{diam_max}], "
        f"C={target_cc}±{cc_tol}"
    )

    for step in range(n_iter):
        if progress_every and step % progress_every == 0:
            print(
                f"progress: {step / n_iter * 100:.1f}% "
                f"accepted={diagnostics['accepted']} "
                f"attempts={diagnostics['proposal_attempts']} "
                f"exact_checked={diagnostics['exact_checked']}"
            )

        result = pick_node_exact(
            mtx=mtx,
            edges=edges,
            non_edges=non_edges,
            degrees=degrees,
            tri_num=tri_num,
            cherry_num=cherry_num,
            target_edges=target_edges,
            target_cc=target_cc,
            cc_tol=cc_tol,
            diam_min=diam_min,
            diam_max=diam_max,
            max_attempts=1,  # one raw proposal per Markov-chain iteration
            diagnostics=diagnostics,
        )

        if result is None:
            # Rejected proposal: remain in the current state. This is a
            # self-transition and the Markov-chain iteration is still counted.
            continue

        if isinstance(result, str):
            diagnostics["stopped_early"] = True
            diagnostics["failed_at_step"] = int(step)
            print(result)
            break

        (
            x,
            y,
            u,
            v,
            idx_e,
            idx_ne,
            tri_num,
            cherry_num,
            verification,
        ) = result

        # Commit accepted move.
        mtx[x, y] = 0
        mtx[y, x] = 0
        mtx[u, v] = 1
        mtx[v, u] = 1

        degrees[x] -= 1
        degrees[y] -= 1
        degrees[u] += 1
        degrees[v] += 1

        edges[idx_e] = [u, v]
        non_edges[idx_ne] = [x, y]

        diagnostics["accepted"] += 1
        diagnostics["last_accepted_verification"] = verification

    if not diagnostics["stopped_early"] and diagnostics["proposal_attempts"] != n_iter:
        raise RuntimeError(
            "Internal error: proposal count does not match n_iter. "
            f"Expected {n_iter}, got {diagnostics['proposal_attempts']}."
        )

    final_verification = exact_validate_graph(
        adj=mtx,
        target_edges=target_edges,
        diam_min=diam_min,
        diam_max=diam_max,
        target_cc=target_cc,
        cc_tol=cc_tol,
    )

    if not final_verification["valid"]:
        raise RuntimeError(
            "Final MCMC state failed exact validation: "
            f"{final_verification}"
        )

    diagnostics["final_verification"] = _json_safe_verification(final_verification)
    diagnostics["final_valid"] = bool(final_verification["valid"])
    diagnostics["completed_iterations"] = int(diagnostics["proposal_attempts"])

    if diagnostics["proposal_attempts"] > 0:
        diagnostics["acceptance_rate"] = (
            diagnostics["accepted"] / diagnostics["proposal_attempts"]
        )
        # Backwards-compatible key retained for existing analysis scripts.
        diagnostics["accepted_per_attempt"] = diagnostics["acceptance_rate"]
    else:
        diagnostics["acceptance_rate"] = 0.0
        diagnostics["accepted_per_attempt"] = 0.0

    if diagnostics["exact_checked"] > 0:
        diagnostics["accepted_per_exact_check"] = (
            diagnostics["accepted"] / diagnostics["exact_checked"]
        )
    else:
        diagnostics["accepted_per_exact_check"] = 0.0

    return mtx, diagnostics
