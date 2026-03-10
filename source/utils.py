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




def process_distances(results_path, min_max=True, weights=None):
    """
    This function processes the distance matrices, normalizes the columns, computes
    the chemical and iface values, and returns the DataFrame.
    
    Parameters:
    - basepath (str): The base directory path where the results are stored.
    - result_dir (str): The relative directory path where distance matrices are stored.
    - min_max (bool): Whether to use min-max normalization (default: True).
    - weights (list, optional): A list of weights for the chemical and iface computations.

    Returns:
    - pd.DataFrame: A processed and reordered DataFrame.
    """

    folders_names = os.listdir(results_path)

    data = []
    for folder in folders_names:
        pair_files_contain = os.listdir(Path(results_path, folder))
        for pair_file_name in pair_files_contain:
            ID1, ID2 = pair_file_name[:-4].split("_")
            pair_file_path = Path(results_path, folder, pair_file_name)
            dist = np.array(np.load(pair_file_path))
            data.append({'ID1': ID1, 'ID2': ID2, f'{folder}': dist})

    # Create DataFrame from collected data
    df = pd.DataFrame(data)

    # Group by ID1 and ID2 and take the first occurrence
    df = df.groupby(['ID1', 'ID2'], as_index=False).first()

    # Normalize columns, keeping unnormalized ones too
    for column in df.columns:
        if column not in ['ID1', 'ID2', 'pair_index']:
            col_max = df[column].max()
            col_min = df[column].min()
            new_column = f'{column}_norm'

            # Apply the chosen normalization method (min-max or max normalization)
            if min_max:
                df[new_column] = (df[column] - col_min) / (col_max - col_min)
            else:
                df[new_column] = df[column] / col_max

    # Compute 'chemical' and 'iface' columns based on the provided weights or default averaging
    if weights:
        df['chemical'] = (weights[1] * df['charge_norm'] + weights[2] * df['hbond_norm'] + weights[3] * df['hphob_norm'])
        df['iface'] = (weights[0] * df['structural_norm'] + weights[1] * df['charge_norm'] + weights[2] * df['hbond_norm'] + weights[3] * df['hphob_norm'])
    else:
        df['chemical'] = (df['charge_norm'] + df['hbond_norm'] + df['hphob_norm']) / 3
        df['iface'] = (df['structural_norm'] + df['charge_norm'] + df['hbond_norm'] + df['hphob_norm']) / 4

    # Start with 'ID1' and 'ID2' as the first two columns
    columns_order = ['ID1', 'ID2']

    # Add 'iface' and 'chemical'
    if 'iface' in df.columns:
        columns_order.append('iface')
    if 'chemical' in df.columns:
        columns_order.append('chemical')

    # Add normalized columns
    normed_columns = [col for col in df.columns if col.endswith('_norm')]
    columns_order.extend(normed_columns)

    # Add original columns at the end
    original_columns = [col for col in df.columns if not col.endswith('_norm') and col not in ['ID1', 'ID2', 'pair_index', 'iface', 'chemical']]
    columns_order.extend(original_columns)

    # Reorder DataFrame columns based on the dynamically generated order
    df = df[columns_order]

    return df

