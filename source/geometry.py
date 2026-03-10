import numpy as np
import trimesh
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import scipy.sparse as sp


from source.config import *



def compute_curvatures(vertices, faces):
    """
    vertices : (N,3) array
    faces    : (M,3) array
    returns  : H (mean), K (gaussian) per vertex
    """
    # PyVista expects faces as: [3, i, j, k, 3, i, j, k, ...]
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    faces_pv = faces_pv.reshape(-1)

    m = pv.PolyData(vertices, faces_pv).triangulate()

    H = np.asarray(m.curvature(curv_type="mean"))
    K = np.asarray(m.curvature(curv_type="gaussian"))

    return H, K





def vertex_average_triangle_area(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute the average area of incident triangles for each vertex using Trimesh.
    """
    n_vertices = len(mesh.vertices)

    # Face areas (M,)
    face_areas = mesh.area_faces

    # Per-vertex face associations (N, max_faces) with -1 padding
    vertex_faces = mesh.vertex_faces  # shape (N, K), padded with -1

    # Initialize average area array
    avg_area = np.zeros(n_vertices, dtype=float)

    # Compute average for each vertex
    for i in range(n_vertices):
        valid_faces = vertex_faces[i]
        valid_faces = valid_faces[valid_faces != -1]  # remove padding
        if len(valid_faces) > 0:
            avg_area[i] = np.mean(face_areas[valid_faces])

    return avg_area



def distance_matrix_shortest_edges_path(mesh_path=None, mesh=None,  verbose=True):
    if verbose and mesh_path:
        print(f"[Dijkstra]: {mesh_path}")
    if mesh_path:    
        mesh = trimesh.load(mesh_path, process=False)

        n = len(mesh.vertices)
        edges = mesh.edges_unique            # (E, 2) int
        lengths = mesh.edges_unique_length   # (E,) float
    if mesh:
        n = len(mesh.vertices)
        edges = mesh.edges_unique            # (E, 2) int
        lengths = mesh.edges_unique_length   # (E,) float

    # Optional: guard against zero-length edges
    lengths = np.asarray(lengths, dtype=np.float64)
    zero = lengths <= 0
    if zero.any():
        if verbose:
            print(f"Found {zero.sum()} zero-length edges; setting tiny epsilon")
        lengths[zero] = np.finfo(np.float64).eps

    # Undirected graph: add both directions
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    data = np.r_[lengths,        lengths      ]

    A = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    if verbose:
        print(f"[dijkstra] graph: V={n}, E={edges.shape[0]} (undirected)")

    # All-pairs shortest paths
    D = dijkstra(csgraph=A, directed=False, return_predecessors=False)

    # If mesh has multiple components, unreachable pairs are inf.
    if not np.isfinite(D).all():
        if verbose:
            comps = len(list(mesh.split(only_watertight=False)))
            print(f"[Dijkstra] warning: found disconnected pairs (components={comps})")

    # Ensure clean symmetry & zeros on diagonal
    D = np.minimum(D, D.T)
    np.fill_diagonal(D, 0.0)
    return D


