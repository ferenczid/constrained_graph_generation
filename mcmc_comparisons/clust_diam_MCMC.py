import numpy as np
import random
import scipy.sparse
import scipy.sparse.csgraph

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

def shuffle_mtx(mtx, swaps=100, tolerance=0.10):
    mtx = mtx.copy()
    
    edges, non_edges = edge_list(mtx)
    degrees = deg_of(mtx) 
    N_tri = triangles(mtx)
    N_cherries = cherries(mtx)
    
    if N_cherries == 0:
        print("Graph has no cherries.")
        return mtx

    targ_cc = N_tri / N_cherries
    

    initial_diam = get_exact_diameter(edges, len(mtx))
    
    d_min = initial_diam * (1 - tolerance)
    d_max = initial_diam * (1 + tolerance)
    
    print(f"Target CC: {targ_cc:.4f}")
    print(f"Exact Initial Diameter: {initial_diam}")
    print(f"Interval: [{d_min:.2f}, {d_max:.2f}]")

    for step in range(swaps):
        if step % 1000 == 0:
            print(f'progress: {step / swaps * 100:.1f}%')
            
        result = pick_node(
            mtx, edges, non_edges, degrees, N_tri, N_cherries, targ_cc, 
            flex=0.025, 
            diam_lower=d_min, 
            diam_upper=d_max
        )
        
        if isinstance(result, str):
            print(result)
            break
            
        x, y, u, v, idx_e, idx_ne, N_tri, N_cherries = result
        

        mtx[x, y] = 0; mtx[y, x] = 0
        mtx[u, v] = 1; mtx[v, u] = 1
        
        degrees[x] -= 1; degrees[y] -= 1
        degrees[u] += 1; degrees[v] += 1
        
        edges[idx_e] = [u, v]
        non_edges[idx_ne] = [x, y]
        
    return mtx

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
