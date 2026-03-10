import os
import numpy as np
import trimesh
import open3d as o3d

def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a Trimesh object.")
    return mesh

# === Normalize Coordinates to RGB ===
def normalize_coords_to_color(coords):
    coords = coords - coords.min(0)
    coords = coords / coords.max(0)
    return (coords * 255).astype(np.uint8)

def trimesh_to_open3d(tri_mesh):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)

    if hasattr(tri_mesh.visual, 'vertex_colors'):
        colors = tri_mesh.visual.vertex_colors[:, :3] / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()
    return mesh

# === Visualizer ===
def show_mesh_in_window(mesh, window_name):
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_name,
        width=800,
        height=600,
        mesh_show_back_face=True
    )

# === Get Top-k Matches and Normalized Weights (coupling matrix) ===
def get_topk_matches_and_weights(soft_map, k=2):
    if k > soft_map.shape[1]:
        raise ValueError(f"Requested top-k={k} exceeds target mesh vertex count {soft_map.shape[1]}")
    topk_idx = np.argsort(soft_map, axis=1)[:, -k:][:, ::-1]
    topk_weights = np.take_along_axis(soft_map, topk_idx, axis=1)
    weight_sums = topk_weights.sum(axis=1, keepdims=True)
    normalized_weights = topk_weights / (weight_sums + 1e-8)
    return topk_idx, normalized_weights

# === Transfer colors using Top-k soft matches (mesh1 -> mesh2) ===
def transfer_colors_topk(mesh1, mesh2, soft_map, k=2):
    topk_indices, weights = get_topk_matches_and_weights(soft_map, k)
    colors1 = normalize_coords_to_color(mesh1.vertices)

    colors2 = np.zeros((len(mesh2.vertices), 3), dtype=np.float32)
    weight_sums = np.zeros(len(mesh2.vertices), dtype=np.float32)

    for src_idx, (targets, ws) in enumerate(zip(topk_indices, weights)):
        for t_idx, w in zip(targets, ws):
            colors2[t_idx] += colors1[src_idx] * w
            weight_sums[t_idx] += w

    valid = weight_sums > 0
    colors2[valid] /= weight_sums[valid, None]
    colors2 = np.clip(colors2, 0, 255).astype(np.uint8)

    return colors1, colors2


def _load_coupling_map_with_reverse_fallback(coupling_matrix_path, mesh1_name, mesh2_name):
    """
    Always map mesh1 -> mesh2. If {mesh1}_{mesh2}.npy not found,
    try {mesh2}_{mesh1}.npy and transpose.
    """
    if os.path.exists(coupling_matrix_path):
        return np.load(coupling_matrix_path), False  # False: not transposed
    # try reverse
    coupling_matrix_dir = os.path.dirname(coupling_matrix_path)
    reverse_path = os.path.join(coupling_matrix_dir, f"{mesh2_name}_{mesh1_name}.npy")
    if os.path.exists(reverse_path):
        return np.load(reverse_path).T, True
    raise FileNotFoundError(
        f"Neither '{coupling_matrix_dir}' nor reverse '{reverse_path}' exist."
    )


def show_color_mapped_meshes(
    mesh1_path,
    mesh2_path,
    coupling_matrix_path=None,
    top_k=2,
):
    """
    Always maps from mesh1 -> mesh2.
    Args:
        mesh1_path, mesh2_path: file paths to meshes.
        coupling_matrix_path: path to npy matrix of shape [V1, V2] (or [V2, V1] if reverse; will be transposed).
        top_k: numbers of top matches for soft map color transfer.
    """
    surf1_name = os.path.splitext(os.path.basename(mesh1_path))[0]
    surf2_name = os.path.splitext(os.path.basename(mesh2_path))[0]

    # === Load meshes ===
    mesh1 = load_mesh(mesh1_path)
    mesh2 = load_mesh(mesh2_path)

    if coupling_matrix_path is None:
        raise ValueError("Coupling matrix path is None.")
    print("Using coupling matrix (with reverse-file fallback if needed).")
    soft_map, transposed = _load_coupling_map_with_reverse_fallback(coupling_matrix_path, surf1_name, surf2_name)
    if transposed:
        print("Loaded reverse soft map and transposed it to get mesh1 -> mesh2.")
    colors1, colors2 = transfer_colors_topk(mesh1, mesh2, soft_map, k=top_k)

    # === Apply colors ===
    mesh1.visual.vertex_colors = np.hstack([colors1, np.full((len(colors1), 1), 255, dtype=np.uint8)])
    mesh2.visual.vertex_colors = np.hstack([colors2, np.full((len(colors2), 1), 255, dtype=np.uint8)])

    # === Convert to Open3D and visualize ===
    o3d_mesh1 = trimesh_to_open3d(mesh1)
    o3d_mesh2 = trimesh_to_open3d(mesh2)

    print("Showing Mesh 1 (colored by XYZ)...")
    show_mesh_in_window(o3d_mesh1, "Mesh 1 (Colored by XYZ)")


    print(f"Showing Mesh 2 (Top-{top_k} Weighted Color Transfer)...")
    show_mesh_in_window(o3d_mesh2, f"Mesh 2 (Top-{top_k} Weighted Color Transfer)")
