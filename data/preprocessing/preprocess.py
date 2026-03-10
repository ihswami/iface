import os
import sys
from pathlib import Path
import shutil
import numpy as np
import trimesh
import trimesh.smoothing as tms 
import pymeshlab
from scipy.spatial import cKDTree
import scipy.sparse as sp

from multiprocessing import Pool, cpu_count


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from source.geometry import distance_matrix_shortest_edges_path as dsep
from source.geometry import compute_curvatures
from source.config import *





def clean_mesh(vertices, faces, min_faces_component=0):
    """
    Clean a mesh:
    - remove invalid faces (out-of-range indices)
    - remove degenerate faces (repeated indices or zero area)
    - optionally remove small connected components (by faces)
    - reindex vertices to remove unreferenced ones
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    n_verts = len(vertices)
    if faces.size == 0:
        return vertices, faces

    # 1) Remove faces with invalid indices
    valid_idx_mask = (faces >= 0) & (faces < n_verts)
    valid_idx_mask = valid_idx_mask.all(axis=1)
    faces = faces[valid_idx_mask]
    if faces.size == 0:
        return vertices, faces

    # 2) Remove degenerate faces (repeated vertex indices)
    f = faces
    deg_repeat = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])

    # 3) Remove near-zero area faces
    v0 = vertices[f[:, 0]]
    v1 = vertices[f[:, 1]]
    v2 = vertices[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area2 = (cross ** 2).sum(axis=1)
    deg_area = area2 < 1e-16

    degenerate = deg_repeat | deg_area
    keep = ~degenerate
    faces = faces[keep]
    if faces.size == 0:
        return vertices, faces

    # 4) Remove small connected components (by faces)
    if min_faces_component > 0:
        n_faces = len(faces)
        vert_to_faces = [[] for _ in range(n_verts)]
        for fi, face in enumerate(faces):
            for vid in face:
                vert_to_faces[int(vid)].append(fi)

        visited = np.zeros(n_faces, dtype=bool)
        components = []
        for start in range(n_faces):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            comp = [start]
            while stack:
                cur = stack.pop()
                face = faces[cur]
                for vid in face:
                    for neigh_face in vert_to_faces[int(vid)]:
                        if not visited[neigh_face]:
                            visited[neigh_face] = True
                            stack.append(neigh_face)
                            comp.append(neigh_face)
            components.append(comp)

        keep_faces_mask = np.zeros(n_faces, dtype=bool)
        for comp in components:
            if len(comp) >= min_faces_component:
                keep_faces_mask[comp] = True

        faces = faces[keep_faces_mask]
        if faces.size == 0:
            return vertices, faces

    # 5) Reindex vertices so we only keep used ones
    used_verts = np.unique(faces.reshape(-1))
    old_to_new = -np.ones(n_verts, dtype=np.int64)
    old_to_new[used_verts] = np.arange(len(used_verts), dtype=np.int64)

    new_vertices = vertices[used_verts]
    new_faces = old_to_new[faces]

    return new_vertices, new_faces


def build_vertex_adjacency(faces, num_vertices):
    i = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1)
    j = faces[:, [1, 0, 2, 1, 0, 2]].reshape(-1)
    data = np.ones(len(i), dtype=np.uint8)
    A = sp.csr_matrix((data, (i, j)), shape=(num_vertices, num_vertices))
    A = ((A + A.T) > 0).astype(np.uint8)
    return A



def k_hop_neighbors(adj_csr, start_idx, hops=1): #BFS
    current = {start_idx}
    visited = set(current)
    for _ in range(hops):
        next_vertices = set()
        for v in current:
            neighbors = adj_csr.indices[adj_csr.indptr[v]:adj_csr.indptr[v+1]]
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    next_vertices.add(n)
        if not next_vertices:
            break
        current = next_vertices
    return np.array(sorted(visited), dtype=np.int64)




def simplify_mesh_and_properties(
    src_dir,
    dest_dir,
    surface,
    target_vertex_count,
    hops=3
):
    mesh_path = os.path.join(src_dir, f"{surface}.ply")

    # 1) Load original mesh and clean 
    mesh = trimesh.load(mesh_path, force='mesh')
    orig_v, orig_f = clean_mesh(mesh.vertices, mesh.faces,
                                      min_faces_component=0)

    mesh_clean = trimesh.Trimesh(vertices=orig_v, faces=orig_f, process=False)
    original_vertices = mesh_clean.vertices
    original_faces    = mesh_clean.faces

    # Build adjacency on original
    adj = build_vertex_adjacency(original_faces, len(original_vertices))

    # 2) Save clean original to temp for pymeshlab decimation
    temp_input  = os.path.join(src_dir, "temp_input.ply")
    temp_output = os.path.join(src_dir, "temp_simplified.ply")
    mesh_clean.export(temp_input)

    # 3) Decimate with pymeshlab 
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_input)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_vertex_count * 2)
    )
    ms.save_current_mesh(temp_output)

    # 4) Load simplified mesh and clean again
    simplified_mesh = trimesh.load(temp_output, force='mesh')
    simp_v, simp_f = clean_mesh(
        simplified_mesh.vertices,
        simplified_mesh.faces,
        min_faces_component=20  # drop tiny islands
    )
    simplified_mesh = trimesh.Trimesh(vertices=simp_v, faces=simp_f, process=False)

    # ---------- Laplacian smoothing ----------
    # tweak lamb and iterations for more/less smoothing
    try:
        tms.filter_laplacian(simplified_mesh, lamb=0.1, iterations=4)
    except Exception as e:
        print(f"Warning: Laplacian smoothing failed: {e}")
    # ----------------------------------------------

    simplified_vertices = simplified_mesh.vertices

    # 5) Map simplified vertices to nearest original vertices
    tree = cKDTree(original_vertices)
    _, idxs = tree.query(simplified_vertices, k=1)
    base_indices = np.atleast_1d(idxs).astype(int)

    os.makedirs(dest_dir, exist_ok=True)

    # 6) Precompute adjacency-based neighborhoods
    neighborhoods = []
    for b in base_indices:
        neigh = k_hop_neighbors(adj, int(b), hops=hops)
        neighborhoods.append(neigh)

    # 7) Simplify all .npy properties
    for fname in os.listdir(src_dir):
        if not fname.endswith(".npy"):
            continue

        prop_path = os.path.join(src_dir, fname)
        prop_data = np.load(prop_path, allow_pickle=True)

        if np.issubdtype(prop_data.dtype, np.number):
            simplified_prop = []
            for neigh in neighborhoods:
                vals = prop_data[neigh]
                simplified_prop.append(vals.mean(axis=0))
            simplified_prop = np.stack(simplified_prop, axis=0)
        else:
            simplified_prop = prop_data[base_indices]

        out_path = os.path.join(dest_dir, fname)
        np.save(out_path, simplified_prop)
        print(f"Saved simplified property: {out_path}")
    # ------ curvature ------
    vertices, faces = simplified_mesh.vertices, simplified_mesh.faces
    H_mean, K_gauss = compute_curvatures(vertices, faces)

    np.save(os.path.join(dest_dir, f"{surface}_mean_curvature.npy"), H_mean.astype(np.float32))
    np.save(os.path.join(dest_dir, f"{surface}_gaussian_curvature.npy"), K_gauss.astype(np.float32))

    # 8) Save final simplified + smoothed mesh
    out_mesh_path = os.path.join(dest_dir, f"{surface}.ply")
    simplified_mesh.export(out_mesh_path)
    print(f"Saved simplified mesh to {out_mesh_path}")

    # 9) Cleanup
    for p in (temp_input, temp_output):
        if os.path.exists(p):
            os.remove(p)


def simplify_mesh_and_properties_wrapper(surface):
    raw_dir = os.path.join(BASEPATH, "data", "raw", surface)
    dest_dir = os.path.join(BASEPATH, "data", "processed", surface)

    mesh_path = os.path.join(raw_dir, f"{surface}.ply")
    if not os.path.exists(mesh_path):
        print(f"Missing mesh for {surface}, skipping.")
        return

    mesh = trimesh.load(mesh_path, force='mesh')
    num_vertices = len(mesh.vertices)

    effective_target = TARGET_VERTEX_COUNT
    if num_vertices > effective_target:
        print(f"Simplifying {surface} from {num_vertices} → {effective_target} vertices...")
        simplify_mesh_and_properties(raw_dir, dest_dir, surface, effective_target)
    
    else:
        os.makedirs(dest_dir, exist_ok=True)  
        for file_name in os.listdir(raw_dir):
            src_file = os.path.join(raw_dir, file_name)
            dest_file = os.path.join(dest_dir, file_name)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dest_file)
        print(f"Copied all files from {raw_dir} to {dest_dir}")



def compute_and_save_shortest_edge_geodesic(name):
    output_dir = os.path.join(BASEPATH, "precomputed_geodesics")
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(BASEPATH, "data", "processed")
    ply_path = os.path.join(data_path, name, f"{name}.ply")
    dist_path = os.path.join(output_dir, f"{name}.npy")

    # Skip if output already exists
    if os.path.exists(dist_path):
        print(f"Skipping {name}: geodesic distance matrix file already exists.")
        return "skipped"

    if not os.path.exists(ply_path):
        print(f"Missing: {ply_path}")
        return "missing"

    try:
        mesh = trimesh.load(ply_path, process=False)

        print(f"Computing geodesic distances for: {name}")
        D = dsep(mesh=mesh)

        np.save(dist_path, D)
        print(f"Saved geodesic distances for {name} to {dist_path}")
        return "ok"

    except Exception as e:
        print(f"Failed {name}: {type(e).__name__}: {e}")
        return "error"


def list_mesh_names(data_path):
    names = []
    for name in os.listdir(data_path):
        ply_path = os.path.join(data_path, name, f"{name}.ply")
        if os.path.isfile(ply_path):
            names.append(name)
    return names



if __name__ == "__main__":
    ncpu = cpu_count()

    raw_path = os.path.join(BASEPATH, "data", "raw")
    des_path = os.path.join(BASEPATH, "data", "processed")

    # ---- simplify ----
    raw_names = list_mesh_names(raw_path)
    workers1 = max(1, ncpu - 1)
    print(f"Simplify: Found {len(raw_names)} raw meshes. Using {workers1} workers.")

    with Pool(processes=workers1) as pool:
        pool.map(simplify_mesh_and_properties_wrapper, raw_names)

    # ----geodesics ----
    names = list_mesh_names(des_path)

    workers2 = max(1, ncpu - 1)
    print(f"Geodesics: Found {len(names)} meshes. Using {workers2} workers.")

    with Pool(processes=workers2) as pool:
        results = pool.map(compute_and_save_shortest_edge_geodesic, names)

