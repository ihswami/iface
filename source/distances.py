import os
import numpy as np
import scipy.sparse as sp
import trimesh


from source.geometry import distance_matrix_shortest_edges_path as dsep
from source.utils import rank_four_distance_difference_tensor_contraction as rftc

from source.config import *


def build_vertex_adjacency(faces, num_vertices):
    i = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1)
    j = faces[:, [1, 0, 2, 1, 0, 2]].reshape(-1)
    data = np.ones(len(i), dtype=np.uint8)
    A = sp.csr_matrix((data, (i, j)), shape=(num_vertices, num_vertices))
    A = ((A + A.T) > 0).astype(np.uint8)
    return A


def kth_ring_indices(adj_csr, start_idx, k=1):
    current = {start_idx}
    visited = {start_idx}
    for _ in range(k):
        next_vertices = set()
        for v in current:
            neighbors = adj_csr.indices[adj_csr.indptr[v]:adj_csr.indptr[v+1]]
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    next_vertices.add(n)
        if not next_vertices:
            return np.array([], dtype=np.int64)
        current = next_vertices
    return np.array(sorted(current), dtype=np.int64)  # exactly k hops



def get_temperature_array(vertices, triangles, k_ring, dist_mat):
    A = build_vertex_adjacency(triangles, len(vertices))
    n = len(vertices)
    global_mean = dist_mat[dist_mat > 0].mean() if (dist_mat > 0).any() else 1.0
    temps = np.empty(n)
    for i in range(n):
        ring = kth_ring_indices(A, i, k_ring)
        if ring.any():
            temps[i] = dist_mat[ring, i].mean()
        else:
            row_nonzero = dist_mat[i][dist_mat[i] > 0]
            temps[i] = row_nonzero.mean() if row_nonzero.size else global_mean
        temps[i] = max(temps[i], 1e-12)
    return temps


def message_passing(field, temps, dist_mat, epsilon=0.1 ):
    temps = np.maximum(temps, 1e-12) 
    weights = np.exp(- dist_mat   / temps[np.newaxis, :])
    return field + epsilon * field @ weights


def globalize_distance(D, sigma, epsilon):
        K = np.exp(-(D**2)/(2*sigma**2))
        K2 = K @ K                      # global mixing
        return D + epsilon * K2



def indices_along_normalized_top_k_weights(matrix, top_k, axis=0):
    """
       Get the top-k entries along the given axis and normalize the selected values
    so each group (column if axis=0, row if axis=1) sums to 1.

    Returns (rank-grouped):
        [
          {group_idx: (weight_at_rank_i, other_axis_index), ...},  # rank 1
          {group_idx: (weight_at_rank_i, other_axis_index), ...},  # rank 2
          ...
          {group_idx: (weight_at_rank_k, other_axis_index), ...},  # rank k
        ]
    """
    n_rows, n_cols = matrix.shape

    if axis == 0:
        k = min(top_k, n_rows)
        sorted_indices = np.argsort(matrix, axis=0)[::-1] # larger to smaller 
        top_indices = sorted_indices[:k, :] # indices values for the top k values in the axis 0
        top_values = np.take_along_axis(matrix, top_indices, axis=0) # values for the above selected indices
        # Normalize so top-k sum to 1 in each column
        col_sums = top_values.sum(axis=0, keepdims=True) 
        norm_values = top_values / col_sums
        # forward map 
        result = [{
            col: (float(norm_values[i, col]), int(top_indices[i, col]))              
            for col in range(n_cols)
        } for i in range(k)]

    elif axis == 1:
        k = min(top_k, n_cols)
        sorted_indices = np.argsort(matrix, axis=1)[:, ::-1]
        top_indices = sorted_indices[:, :k]
        top_values = np.take_along_axis(matrix, top_indices, axis=1)
        # Normalize so top-k sum to 1 in each row
        row_sums = top_values.sum(axis=1, keepdims=True)
        norm_values = top_values / row_sums
        result = [{
            row: (float(norm_values[row, i]), int(top_indices[row, i]))
            for row in range(n_rows)
        } for i in range(k)]

    else:
        raise ValueError("Axis must be 0 (per column) or 1 (per row)")

    return result



# weighted distances

def compute_weighted_hard_distance( feature_name, surface1_name, surface2_name,  T=None, map12=None, map21=None):
    """
    Compute a weighted 'hard' correspondence distance between two surfaces.

    Conventions:
      - T is (m, n): rows correspond to points of surface1, cols to points of surface2.
      - map12 is length m: for each i in surface1, map12[i] gives index j in surface2.
      - map21 is length n: for each j in surface2, map21[j] gives index i in surface1.

    Args:
        T: np.ndarray, shape (m, n), coupling matrix.
        feature_name: str, 'geometry' or name of precomputed feature (e.g., 'shot', 'xyz', ...).
        surface1_name, surface2_name: str, identifiers used to locate data on disk.
        map12, map21: optional index arrays. If either is None, they are derived from T via argmax.

    Returns:
        list of bidirectional distances.
    """

    # ---- Basic checks on T ----
    if not isinstance(T, np.ndarray) or T.ndim != 2:
        raise ValueError("T must be a 2D numpy array of shape (m, n).")
    if not np.all(np.isfinite(T)):
        raise ValueError("T contains non-finite values.")
    m, n = T.shape

    mesh1_path = os.path.join(BASEPATH, 'data', 'processed', surface1_name, f'{surface1_name}.ply')
    mesh2_path = os.path.join(BASEPATH, 'data', 'processed', surface2_name, f'{surface2_name}.ply')
    geo_path1 = os.path.join(BASEPATH, 'precomputed_geodesics', f'{surface1_name}.npy')
    geo_path2 = os.path.join(BASEPATH, 'precomputed_geodesics', f'{surface2_name}.npy')

    if os.path.exists(geo_path1):
        D1 = np.load(geo_path1)
    else:
        D1 = dsep(mesh1_path)

    if os.path.exists(geo_path2):
        D2 = np.load(geo_path2) 
    else:
        D2 = dsep(mesh2_path)       
    
    mesh1 = trimesh.load(mesh1_path, process=False)
    mesh2 = trimesh.load(mesh2_path, process=False)

    # Extract vertices and triangle faces
    vert1, tri1 = mesh1.vertices, mesh1.faces
    vert2, tri2 = mesh2.vertices, mesh2.faces
    

    def _validate_or_build_maps(map12, map21):
        if (map12 is None) or (map21 is None):
            # complete this part when only T is given
            m12 = np.argmax(T, axis=1)  # (m,)
            m21 = np.argmax(T, axis=0)  # (n,)
                
        else:
            m12_weights, m12 = np.array(list(map12.values())).T
            m21_weights, m21 = np.array(list(map21.values())).T
            m12 = m12.astype(int)
            m21 = m21.astype(int)
            if m12.shape != (m,):
                raise ValueError(f"map12 must have shape ({m},), got {m12.shape}.")
            if m21.shape != (n,):
                raise ValueError(f"map21 must have shape ({n},), got {m21.shape}.")
        # bounds check
        if (m12.min() < 0) or (m12.max() >= n):
            raise IndexError("map12 contains indices outside [0, n).")
        if (m21.min() < 0) or (m21.max() >= m):
            raise IndexError("map21 contains indices outside [0, m).")
        
        return m12, m12_weights, m21, m21_weights

    if feature_name != 'structural':
        # ---- Load features ----
        feat1_path = os.path.join(BASEPATH, 'data', 'processed', surface1_name, f'{surface1_name}_{feature_name}.npy')
        feat2_path = os.path.join(BASEPATH, 'data', 'processed', surface2_name, f'{surface2_name}_{feature_name}.npy')


        feat1 = np.load(feat1_path) # (m,) or (m, d)
        feat2 = np.load(feat2_path)  # (n,) or (n, d)
        
   
        fm1 = 0
        fm2 = 0
        for k in [2]:
            tem1 = get_temperature_array(vert1, tri1, k_ring=k, dist_mat= D1)
            tem2 = get_temperature_array(vert2, tri2, k_ring=k, dist_mat= D2)
            
            fm1 +=  message_passing(feat1, tem1, D1, epsilon=0.1 ) 
            fm2 +=  message_passing(feat2, tem2, D2, epsilon=0.1 )
        feat1 = fm1 
        feat2 = fm2 

        # Ensure 2D (treat scalars as 1D features)
        if feat1.ndim == 1:
            feat1 = feat1[:, None]
        if feat2.ndim == 1:
            feat2 = feat2[:, None]

        if feat1.shape[0] != m or feat2.shape[0] != n:
            raise ValueError(
                f"Feature/transport size mismatch: T is ({m},{n}), "
                f"feat1 has {feat1.shape[0]} rows, feat2 has {feat2.shape[0]} rows."
            )

        # ---- Maps ----
        map12, map12_weights, map21, map21_weights = _validate_or_build_maps(map12, map21)
        feat1_mapped = feat2[map12]      # (m, d): best match in surface2 for each point in surface1
        feat2_mapped = feat1[map21]  # (n, d): best match in surface1 for each point in surface2

        # compute weighted L1 distance
        d21 = np.sum((((abs(feat1.flatten()  - 
                        feat1_mapped.flatten()))) )  * map12_weights, axis=0).sum() 
        d12 = np.sum((((abs(feat2.flatten() - 
                        feat2_mapped.flatten()))) )  * map21_weights, axis=0).sum() 
        
        
        dist = (d12, d21)

    else:
        # ---- Structural branch ----

        map12, map12_weights, map21, map21_weights = _validate_or_build_maps(map12, map21)
    
        T12 = np.zeros_like(T, dtype=float)
        # surface1 -> surface2
        T12[np.arange(m), map12] = map12_weights  
    
        
        DG1 = globalize_distance(D1, sigma= 0.1 * np.max(D1), epsilon= 0.1)
        DG2 = globalize_distance(D2, sigma= 0.1 * np.max(D2), epsilon= 0.1)
        
        # surface1 ->  surface2
        dist12 = float((rftc(DG1, DG2, T12)))


        T21 = np.zeros_like(T, dtype=float)
        
        # surface2 ->  surface1
        T21[map21, np.arange(n)] = map21_weights  
       
        dist21 = float((rftc(DG1, DG2, T21)))      
       
        dist = (dist12, dist21)

    return dist




def compute_feature_distance(T, feature, surface1_name, surface2_name, top_k=1):
    forward = indices_along_normalized_top_k_weights(T, axis=1, top_k=top_k)
    backward = indices_along_normalized_top_k_weights(T, axis=0, top_k=top_k)
    
    dist_list = []
    for k in range(top_k):    
        d1, d2 = compute_weighted_hard_distance(feature_name=feature, surface1_name=surface1_name, surface2_name= surface2_name, 
                                                T=T,  map12 = forward[k], map21 = backward[k])
        dist_list.append(d1)
        dist_list.append(d2)

    if feature == 'geometry':
        dist_list_sum = np.sqrt(0.5 * np.array(dist_list).sum())    
    else:
        dist_list_sum = 0.5 * np.array(dist_list).sum()

    dist = dist_list_sum

    return dist   

