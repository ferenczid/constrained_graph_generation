# metrics_utils.py
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import shortest_path


def edge_count_from_adj(adj: np.ndarray) -> int:
    return int(np.sum(adj) // 2)


def transitivity_from_adj(adj: np.ndarray) -> float:
    deg = np.sum(adj, axis=1)
    denom = np.sum(deg * (deg - 1))

    if denom == 0:
        return 0.0

    numer = np.sum(adj * (adj @ adj))
    return float(numer / denom)


def exact_diameter_from_adj(adj: np.ndarray):
    csr = scipy.sparse.csr_matrix(adj)
    dists = shortest_path(csr, directed=False, unweighted=True)

    if np.isinf(dists).any():
        return np.inf

    return int(np.max(dists))


def exact_validate_graph(
    adj: np.ndarray,
    target_edges: int,
    diam_min: int,
    diam_max: int,
    target_cc: float,
    cc_tol: float,
):
    m = edge_count_from_adj(adj)
    cc = transitivity_from_adj(adj)
    diam = exact_diameter_from_adj(adj)

    m_ok = m == target_edges
    cc_ok = abs(cc - target_cc) <= cc_tol
    diam_ok = diam_min <= diam <= diam_max

    return {
        "valid": bool(m_ok and cc_ok and diam_ok),
        "edge_count": m,
        "transitivity": cc,
        "diameter": diam,
        "m_ok": bool(m_ok),
        "cc_ok": bool(cc_ok),
        "diam_ok": bool(diam_ok),
    }
