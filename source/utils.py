import os
import numpy as np
from pathlib import Path
import pandas as pd
import scipy.sparse as sp

from source.config import *


def rank_four_distance_difference_tensor_contraction(D1, D2, T):
    # prepare T as sparse matrix
    T = sp.csr_matrix(T)
    # row/col sums
    row_sum = np.asarray(T.sum(axis=1)).ravel()
    col_sum = np.asarray(T.sum(axis=0)).ravel()

    # skip rows/cols where T is completely zero
    nz_row = row_sum != 0
    nz_col = col_sum != 0

    D1_nz = D1[nz_row][:, nz_row]
    D2_nz = D2[nz_col][:, nz_col]

    r = row_sum[nz_row]
    c = col_sum[nz_col]

    # term1: sum_{ij} r_i r_j D1_ij^2
    term1 = np.sum(D1_nz**2 * r[:, None] * r[None, :])

    # term2: sum_{ij} c_i c_j D2_ij^2
    term2 = np.sum(D2_nz**2 * c[:, None] * c[None, :])

    # term3: -2 * sum( (T^T D1 T) * D2 )
    temp = T.T @ (D1 @ T)
    term3 = -2.0 * np.sum(temp * D2)

    return term1 + term2 + term3



def save_coupling_matrix(T, surface1_name, surface2_name, output_dir=None):
    """
    Save the coupling matrix to a .npy file.
    """
    if output_dir == None:
        output_dir = os.path.join(BASEPATH, 'results', 'coupling_matrix')
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{surface1_name}_{surface2_name}.npy'
    save_path = os.path.join(output_dir, filename)

    np.save(save_path, T)
    print(f"[INFO] Saved coupling matrix to: {save_path}")


def save_results(distance=None, label='label', surface1_name='surf1', surface2_name='surf2', output_dir=None ):
    """
    Save distances to a .npy file.
    """
    if output_dir == None:
        output_dir = os.path.join(BASEPATH, 'results', 'distances', f'{label}')
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{surface1_name}_{surface2_name}.npy'
    save_path = os.path.join(output_dir, filename)

    np.save(save_path, distance)
    print(f"[INFO] Saved {label} distance to: {save_path}")

    
def get_saved_file_path(label, surface1_name, surface2_name):
    if label == 'coupling_matrix':
        output_dir = os.path.join(BASEPATH, 'results', 'coupling_matrix')
        filename1 = f'{surface1_name}_{surface2_name}.npy'
        filename2 = f'{surface2_name}_{surface1_name}.npy'
        
        save_path1 = os.path.join(output_dir, filename1)
        save_path2 = os.path.join(output_dir, filename2)
        save_path = [save_path1, save_path2]
        return save_path


    else:        
        output_dir = os.path.join(BASEPATH, 'results', 'distances', f'{label}')
        filename1 = f'{surface1_name}_{surface2_name}.npy'
        filename2 = f'{surface2_name}_{surface1_name}.npy'
        
        save_path1 = os.path.join(output_dir, filename1)
        save_path2 = os.path.join(output_dir, filename2)
        save_path = [save_path1, save_path2]
        
        return save_path



def build_vertex_adjacency(faces, num_vertices):
    i = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1)
    j = faces[:, [1, 0, 2, 1, 0, 2]].reshape(-1)
    data = np.ones(len(i), dtype=np.uint8)
    A = sp.csr_matrix((data, (i, j)), shape=(num_vertices, num_vertices))
    A = ((A + A.T) > 0).astype(np.uint8)
    return A



def compute_normalization_ranges(vertex_count):
    scale_ratio = vertex_count / TARGET_VERTEX_COUNT

    scaled_minimums = {
        feature: REF_MIN[feature] * scale_ratio ** P_EXPONENTS[feature]
        for feature in P_EXPONENTS
    }

    scaled_maximums = {
        feature: REF_MAX[feature] * scale_ratio ** P_EXPONENTS[feature]
        for feature in P_EXPONENTS
    }

    return scaled_minimums, scaled_maximums



def get_normalization_ranges(vertex_count):
    if vertex_count == 3000:
        return REF_MIN.copy(), REF_MAX.copy()

    return compute_normalization_ranges(vertex_count)
    

def process_distances(results_path, weights=None):
    """
    Load pairwise distances, apply fixed range normalization,
    and compute geometry, chemical, and IFACE distances.
    """
    if weights is None:
        weights = [1, 1, 1, 1, 1]

    if len(weights) != 5:
        raise ValueError("weights must contain exactly five values.")

    data = []

    for folder_path in Path(results_path).iterdir():
        if not folder_path.is_dir():
            continue

        feature = folder_path.name

        for pair_file_path in folder_path.glob("*.npy"):
            try:
                ID1, ID2 = pair_file_path.stem.split("_")
            except ValueError as exc:
                raise ValueError(
                    f"Unexpected filename: {pair_file_path.name}. "
                    "Expected ID1_ID2.npy."
                ) from exc

            try:
                distance = np.asarray(
                    np.load(pair_file_path, allow_pickle=False)
                ).squeeze()

                if distance.size != 1:
                    raise ValueError("Distance file must contain one value.")

                data.append({
                    "ID1": ID1,
                    "ID2": ID2,
                    feature: float(distance),
                })

            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load {pair_file_path}"
                ) from exc

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError(f"No distance files found in {results_path}.")

    df = df.groupby(["ID1", "ID2"], as_index=False).first()

    # Compute the fixed ranges once.
    reference_minimums, reference_maximums = (
        get_normalization_ranges(TARGET_VERTEX_COUNT)
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in {"ID1", "ID2", "pair_index"}
    ]

    for column in feature_columns:
        if column not in reference_minimums:
            raise KeyError(
                f"No normalization range defined for '{column}'."
            )

        distance_range = (
            reference_maximums[column]
            - reference_minimums[column]
        )

    

        df[f"{column}_norm"] = df[column] / distance_range

    required_columns = {
        "structural_norm",
        "mean_curvature_norm",
        "charge_norm",
        "hbond_norm",
        "hphob_norm",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing required distance columns: {sorted(missing_columns)}"
        )

    df["geometry"] = (
        weights[0] * df["structural_norm"]
        + weights[1] * df["mean_curvature_norm"]
    )

    df["chemical"] = (
        weights[2] * df["charge_norm"]
        + weights[3] * df["hbond_norm"]
        + weights[4] * df["hphob_norm"]
    )

    df["iface"] = df["geometry"] + df["chemical"]

    normalized_columns = [
        column for column in df.columns
        if column.endswith("_norm")
    ]

    original_columns = [
        column for column in feature_columns
        if column in df.columns
    ]

    columns_order = (
        ["ID1", "ID2", "iface", "geometry", "chemical"]
        + normalized_columns
        + original_columns
    )

    return df[columns_order]

