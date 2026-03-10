import os
import time
import numpy as np
import ot
import trimesh
import open3d as o3d
from scipy.spatial.distance import cdist
from cycpd import deformable_registration

from source import geometry as geo
from source.config import *


def project_to_transport(T, r, c, iters=100_000, tol=1e-9, verbose=False):
    """
    Project matrix T onto the set of matrices with row sums r and column sums c
    using Sinkhorn-Knopp iterations.

    Convergence is checked using marginal errors.
    """

    T = np.asarray(T, float)
    r = np.asarray(r, float)
    c = np.asarray(c, float)

   
    # CHECK MARGINAL COMPATIBILITY

    if abs(r.sum() - c.sum()) > 1e-12:
        raise ValueError("r and c must sum to the same value")

    # REGULARIZE TO AVOID DIVISION BY ZERO
   
    K = np.maximum(T, 1e-12)

    # INITIALIZE SCALING VECTORS

    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])

    # SINKHORN ITERATIONS
    
    for it in range(iters):

        # Update scalings
        u = r / (K @ v)
        v = c / (K.T @ u)

        # Reconstruct transport matrix
        P = K * u[:, None] * v[None, :]

        # MARGINAL-BASED CONVERGENCE CHECK 
    
        row_error = np.linalg.norm(P.sum(axis=1) - r, 1)
        col_error = np.linalg.norm(P.sum(axis=0) - c, 1)

        if row_error < tol and col_error < tol:
            if verbose:
                print(f" Converged via marginals at iteration {it}")
                print(f" Row error: {row_error:.3e}")
                print(f" Col error: {col_error:.3e}")
            break

    else:
        if verbose:
            print(" Warning: Sinkhorn did NOT converge within max iterations.")
            print(f" Final row error: {row_error:.3e}")
            print(f" Final col error: {col_error:.3e}")

    # FINAL TRANSPORT MATRIX

    return P


def estimate_voxel_size(pcd: o3d.geometry.PointCloud,
                        multiplier: float = 3.0) -> float:
    """
    Estimate voxel size from typical point spacing.
    """
    if len(pcd.points) < 50:
        raise ValueError("Point cloud too small to estimate spacing reliably.")

    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    dists = dists[np.isfinite(dists)]
    if len(dists) == 0:
        raise ValueError("Could not compute nearest-neighbor distances.")

    spacing = np.median(dists)
    return float(multiplier * spacing)



def init_transport_plan(
    src_path,
    tgt_path,
    BASEPATH,
    surface1_name,
    surface2_name,
    feature_list,
    n_samples_align=2500,
    eps=0.01,
    color_weight=1.0,          
    maxiter_cpd=1000,
    tol_cpd=1e-5,
    verbose=False
):
    if verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # -----------------------
    # helpers
    # -----------------------
    def unitnorm_mesh(m: o3d.geometry.TriangleMesh):
        V = np.asarray(m.vertices, dtype=np.float64)
        V = V - V.mean(0)
        s = np.linalg.norm(V, axis=1).max()
        if s > 0:
            V = V / s
        m.vertices = o3d.utility.Vector3dVector(V)
        return m


    def fpfh(pc, radius_normal, radius_feature, max_nn_normal=30, max_nn_feature=100):
        pc = pc  
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal)
        )
        return o3d.pipelines.registration.compute_fpfh_feature(
            pc,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_feature),
        )


    def load_fields(base, name, feature_list):
        p = os.path.join
        fields = []
        for feat in feature_list:
            field = np.load(p(base, "data", 'processed', name, f"{name}_{feat}.npy"))
            fields.append(field)

        return fields
    
    def norm_pair(f1, f2, eps=1e-8):
        """Joint min-max to [-1,1] for both arrays."""
        allv = np.concatenate([np.asarray(f1).ravel(), np.asarray(f2).ravel()])
        vmin = float(allv.min())
        vmax = float(allv.max())
        rng = (vmax - vmin) + eps
        f1n = (f1 - vmin) / rng
        f2n = (f2 - vmin) / rng
        # Shift to [-1, 1]
        f1n = 2 * f1n - 1
        f2n = 2 * f2n - 1
        return f1n, f2n

    def norm_fields_pair(fields1, fields2, eps=1e-8):
        normed1 = []
        normed2 = []
        for f1, f2 in zip(fields1, fields2):
            f1n, f2n = norm_pair(f1, f2, eps)
            normed1.append(f1n)
            normed2.append(f2n)
        return normed1, normed2
    # -----------------------
    # load + normalize meshes
    # -----------------------
    src_mesh = unitnorm_mesh(o3d.io.read_triangle_mesh(src_path))
    tgt_mesh = unitnorm_mesh(o3d.io.read_triangle_mesh(tgt_path))

    n_src = np.asarray(src_mesh.vertices).shape[0]
    n_tgt = np.asarray(tgt_mesh.vertices).shape[0]
    n_samples_align = int(min(n_samples_align, n_src, n_tgt))

    # -----------------------
    # coarse rigid alignment (FPFH+RANSAC+ICP) on DOWN-SAMPLED points
    # -----------------------
    src_s = src_mesh.sample_points_poisson_disk(n_samples_align)
    tgt_s = tgt_mesh.sample_points_poisson_disk(n_samples_align)

  
    voxel_size = estimate_voxel_size(src_s, multiplier=3.0)
    if verbose:
        print(" Estimated voxel_size:", voxel_size)


    # These should be tied together
    radius_normal  = 2.0 * voxel_size
    radius_feature = 5.0 * voxel_size
    max_corr_ransac = 1.5 * voxel_size

    src_fpfh = fpfh(src_s, radius_normal, radius_feature)
    tgt_fpfh = fpfh(tgt_s, radius_normal, radius_feature)

    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_s, tgt_s,
        src_fpfh, tgt_fpfh,
        mutual_filter=False,
        max_correspondence_distance=max_corr_ransac,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.75),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr_ransac),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999)
    )

    # ICP: threshold typically similar order to voxel_size
    icp_threshold = 2.0 * voxel_size
    reg_icp = o3d.pipelines.registration.registration_icp(
        src_s, tgt_s,
        icp_threshold,
        res.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )

    T_coarse = reg_icp.transformation
    # -----------------------
    # full resolution point clouds (geometry)
    # -----------------------
    src_full = o3d.geometry.PointCloud()
    src_full.points = src_mesh.vertices
    tgt_full = o3d.geometry.PointCloud()
    tgt_full.points = tgt_mesh.vertices
    voxel_size_S = estimate_voxel_size(src_full)
    voxel_size_T = estimate_voxel_size(tgt_full)

    radius_normal  = [2.0 * voxel_size_S, 2.0 * voxel_size_T]
    i =0 
    for p in (src_full, tgt_full):    
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal[i], max_nn=100))
        i+=1
    
    icp_threshold = 2.0 * voxel_size_S

    reg_icp_fine = o3d.pipelines.registration.registration_icp(
        src_full, tgt_full,
        icp_threshold,
        T_coarse, # transformation matrix 4X4
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500),
    )
    Trigid = reg_icp_fine.transformation
    src_full.transform(Trigid)

    X_geom = np.asarray(src_full.points, dtype=np.float64)  # (n_s, 3), rigid-aligned source
    Y_geom = np.asarray(tgt_full.points, dtype=np.float64)  # (n_t, 3)
    n_s, n_t = X_geom.shape[0], Y_geom.shape[0]

    ################################################################################
    # Fields are in same order as feature_list
    fields1 = load_fields(BASEPATH, surface1_name, feature_list)
    fields2 = load_fields(BASEPATH, surface2_name, feature_list)
    normed_fields1, normed_fields2 = norm_fields_pair(fields1, fields2)
    
    ################################################################################
    feat1 = np.stack(normed_fields1, axis=1)  # (n_s, d)
    feat2 = np.stack(normed_fields2, axis=1)  # (n_t, d)
    if verbose:
        print(f" Feature matrix shapes: feat1={feat1.shape}, feat2={feat2.shape}")
    # safety if sizes mismatch for some reason
    feat1 = feat1[:n_s]
    feat2 = feat2[:n_t]

    X_feat = np.concatenate([X_geom, color_weight * feat1], axis=1)  # (n_s, 6)
    Y_feat = np.concatenate([Y_geom, color_weight * feat2], axis=1)  # (n_t, 6)
  
    # ---- feature-real space non-rigid CPD ----
    reg = deformable_registration(
        X=Y_feat, #target
        Y=X_feat, #source: source moves to align to target
        max_iterations=float(maxiter_cpd),
        tolerance=float(tol_cpd),
        sigma2=None,
        low_rank=True,
        verbose=verbose,
    )
    TY_feat, _ = reg.register() 

    C = cdist(TY_feat, Y_feat, metric="sqeuclidean")  # (n_s, n_t)
    K = np.exp(-C / max(eps**2, 1e-8))   
    return K


def check_array(arr, name="array", nonneg=False, prob=False, tol=1e-8, verbose=False):
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f" [ERROR] {name}: empty"); return False
    if np.isnan(arr).any():
        print(f" [ERROR] {name}: NaN found"); return False
    if np.isinf(arr).any():
        print(f" [ERROR] {name}: Inf found"); return False
    if nonneg and (arr < 0).any():
        print(f" [ERROR] {name}: negatives found"); return False
    if prob:
        if arr.ndim != 1:
            print(f" [ERROR] {name}: must be 1D for probability vector"); return False
        if (arr < 0).any():
            print(f" [ERROR] {name}: negatives in prob"); return False
        s = arr.sum()
        if not np.isclose(s, 1.0, atol=tol):
            print(f" [WARN] {name}: sum={s:.6f}, not 1.0")
    if verbose:        
        print(f" [OK] {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
    return True


def check_clean_array(arr, name="array", nonneg=False, prob=False, tol=1e-8, verbose=False):
    arr = np.asarray(arr, dtype=float)  # ensure float for NaN/Inf handling

    if arr.size == 0:
        print(f" [ERROR] {name}: empty")
        return False

    # Replace NaN and Inf with 0
    if np.isnan(arr).any() or np.isinf(arr).any():
        print(f" [INFO] {name}: replacing NaN/Inf with 0")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Check nonnegativity if requested
    if nonneg and (arr < 0).any():
        print(f" [ERROR] {name}: negatives found")
        return False

    # Check probability vector conditions
    if prob:
        if arr.ndim != 1:
            print(f" [ERROR] {name}: must be 1D for probability vector")
            return False
        if (arr < 0).any():
            print(f" [ERROR] {name}: negatives in prob")
            return False
        s = arr.sum()
        if not np.isclose(s, 1.0, atol=tol):
            print(f" [WARN] {name}: sum={s:.6f}, not 1.0")
    if verbose:
     print(f" [OK] {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
    return arr  # return cleaned array too


def unitnorm_mesh(m):
    V = m.vertices
    V = V - V.mean(0)
    s = np.linalg.norm(V, axis=1).max()
    if s > 0:
        V = V / s
    m.vertices = V
    return m


def get_marginal_distribution(surface1_name, surface2_name,
                               feature_list = ['charge', 'hphob', 'hbond', 'mean_curvature'],
                                area_only=False):
    surface1_path = os.path.join(BASEPATH, 'data', 'processed', surface1_name, f'{surface1_name}.ply')
    mesh1 = trimesh.load(surface1_path)
    surface2_path = os.path.join(BASEPATH, 'data', 'processed', surface2_name, f'{surface2_name}.ply')
    mesh2 = trimesh.load(surface2_path)

    # vertex-based area weight (not normalized)
    a = geo.vertex_average_triangle_area(mesh1)   
    b = geo.vertex_average_triangle_area(mesh2)   

    weights1 = a / a.sum()
    weights2 = b / b.sum()
    
    if area_only:
        return weights1, weights2
    
    for feat_name in feature_list:
        feat_path1 = os.path.join(BASEPATH, 'data', 'processed', surface1_name, f'{surface1_name}_{feat_name}.npy')
        feat1 = np.load(feat_path1)
        feat_path2 = os.path.join(BASEPATH, 'data', 'processed', surface2_name, f'{surface2_name}_{feat_name}.npy')
        feat2 = np.load(feat_path2)
        featmin = min(feat1.min(), feat2.min())
        # make it non-negative but do NOT normalize
        
        feat1 = (feat1 - featmin) * a
        feat1 = feat1 / feat1.sum()
        weights1 += feat1 

        feat2 = (feat2 - featmin) * b
        feat2 = feat2 / feat2.sum()
        weights2 += feat2 

    # normalize final result
    weights1 /= weights1.sum()
    weights2 /= weights2.sum()


    return weights1, weights2


def optimize_coupling(surface1_name, surface2_name, feature_list,  M, verbose=False):
    print(" === Starting optimzation for coupling matrix ===")
    
    # Paths
    surface1_path = os.path.join(BASEPATH, 'data', 'processed', surface1_name, f'{surface1_name}.ply')
    surface2_path = os.path.join(BASEPATH, 'data', 'processed', surface2_name, f'{surface2_name}.ply')
    
    
    mesh1 = trimesh.load(surface1_path)
    mesh2 = trimesh.load(surface2_path)
  
    mesh1 = unitnorm_mesh(trimesh.load(surface1_path))
    mesh2 = unitnorm_mesh(trimesh.load(surface2_path))

    if verbose:
        print(" Computing geodesic matrices via shortest edge paths...")
    # Geodesic matrices
    C1 = geo.distance_matrix_shortest_edges_path(mesh=mesh1, verbose=verbose)
    C2 = geo.distance_matrix_shortest_edges_path(mesh=mesh2, verbose=verbose)
    
    def globalize_distance(D, sigma, epsilon):
        """ Make distance matrix more global via kernel smoothing. """
        K2 = 0
        for sigma in sigma:
            K = np.exp(-(D**2)/(2*sigma**2))
            K2_int = K @ K 
            K2 += K2_int
                                    # global mixing
        return D + epsilon * K2
    
    C1 = globalize_distance(C1, sigma=[C1.max() * 0.1], epsilon=0.1)
    C2 = globalize_distance(C2, sigma=[C2.max() * 0.1],  epsilon=0.1)
    
    if verbose:
        print(f"C1 shape: {C1.shape}, C2 shape: {C2.shape}")
    
     # cheack initial inputs
    C1 = check_clean_array(C1, "C1", nonneg=True, verbose=verbose)    
    C2 = check_clean_array(C2, "C2", nonneg=True, verbose=verbose)


    max_M = np.max(M)
    M = M /max_M
   
    a, b = get_marginal_distribution(surface1_name=surface1_name, surface2_name=surface2_name, 
                                     feature_list = feature_list,  
                                     area_only=False)
    
   
    
    # cheack inputs
    check_array(a, "a", nonneg=True, prob=True, verbose=verbose) 
    check_array(b, "b", nonneg=True, prob=True, verbose=verbose)
    check_array(M, "M", nonneg=True, verbose=verbose)    
    check_array(C1, "C1", nonneg=True, verbose=verbose)    
    check_array(C2, "C2", nonneg=True, verbose=verbose)

    # objective function optimization
    if verbose:
        print(" Starting optimization...")
    
    start_time = time.time()
    
    start_time1 = time.time()
    
    end_time1 = time.time()
    elapsed_time =  end_time1 - start_time1

    start_time = time.time()    
    def clip_rescale(X, q=0.99):
        X = X.astype(float)
        X = np.maximum(X, 0)                   # no negatives
        cap = np.quantile(X, q)
        X = np.minimum(X, cap)                 # clip big outliers
        s = np.median(X[X>0]) if np.any(X>0) else X.max()
        s = s if np.isfinite(s) and s>0 else 1.0
        return X / s

    C1s1 = clip_rescale(C1)
    C2s1 = clip_rescale(C2)
    Ms  = clip_rescale(M)     # for feature cost
    
    def safe_epsilon(*costs, safety=700.0, base=0.05):
        maxc = max(float(c.max()) for c in costs)
        eps_min = maxc / safety               # avoid underflow
        # also not too small: ~5% of a typical cost scale
        eps_base = base * np.median([np.median(c[c>0]) if np.any(c>0) else maxc for c in costs])
        return max(eps_min, eps_base, 1e-6)
    
    epsilon = safe_epsilon(C1s1, C2s1, Ms)

    print(" ==== Estimating the initial coupling matrix ====")
    init = init_transport_plan(surface1_path, surface2_path, BASEPATH=BASEPATH, surface1_name=surface1_name,
                            surface2_name=surface2_name, feature_list=feature_list, color_weight=0.1, verbose=verbose)
    G0 = init

    G0 = project_to_transport(T=G0, r= a, 
                                c=b, tol=1e-4, verbose=verbose)

    print(" ==== Optimizing objective function ====")
    if verbose:
        print(" Running optimization using entropic regularization...")
    T0 = ot.gromov.entropic_fused_gromov_wasserstein(
    Ms, C1s1, C2s1, a, b,
    epsilon=epsilon,
    alpha=0.9,    
    max_iter=50,
    tol=1e-9,
    armijo=False,
    G0=G0,
    verbose=verbose,
    )
    T0 = project_to_transport(T=T0, r= a, c=b, tol=1e-8, verbose=verbose)
    if verbose:
        print(' End of entropic optimization')
    
        print(' Running non-entropic optimization to refine the solution...')  
            
    T0 = ot.gromov.fused_gromov_wasserstein(
            M, C1, C2, a, b,
            alpha=0.9,
            armijo=False,
            G0=T0,
            verbose=verbose
                )
    end_time1 = time.time()
    elapsed_time =  end_time1 - start_time1
    if verbose:
        print(f" Non-entropic optimization step took {elapsed_time:.2f} seconds")

    
  

    end_time = time.time()
    
   
    if verbose:
        print(f" Optimization step took {end_time - start_time:.2f} seconds")
        print(f" T0 shape: {T0.shape}, sum={T0.sum():.6f}")
    print(" === Finished optimization ===")

    return T0





