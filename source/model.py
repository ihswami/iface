import os
import numpy as np
import ot

from source import optim
from source.utils import  save_coupling_matrix, save_results, get_saved_file_path
from source.distances import compute_feature_distance

from source.config import *

class IFACE:

    def __init__(self, surface1_name, surface2_name,
                 features_list = ['charge', 'hphob', 'hbond', 'mean_curvature'],
                 exact_only=False,
                 verbose=False
                  ):
        self.surface1_name = surface1_name
        self.surface2_name = surface2_name
        self.features_list = features_list
        self.verbose = verbose
        
                    
    def compute(self):
        if self.surface1_name == self.surface2_name:
            print("Both surfaces are the same -- skipping computation.")
            return
      
        M = self.scalar_field_cost(self.surface1_name, self.surface2_name)
        coupling_paths = get_saved_file_path(
            label='coupling_matrix',
            surface1_name=self.surface1_name,
            surface2_name=self.surface2_name
        )

        # Load from available paths
        for path in coupling_paths:
            if os.path.exists(path):
                try:
                    T = np.load(path)
                    if self.verbose:
                        print(f"Loaded coupling matrix from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
        else:
            if self.verbose:
                print("coupling matrix not found -- computing.")
            T = optim.optimize_coupling(self.surface1_name, self.surface2_name, self.features_list, M, self.verbose)            
            save_coupling_matrix(T, self.surface1_name, self.surface2_name)

        
        # Loop over all features + geometry
        for feature in self.features_list + ['structural']:
            if feature not in ('mean_curvature', 'gaussian_curvature'):
                check_feature_paths = get_saved_file_path(
                    label=feature,
                    surface1_name=self.surface1_name,
                    surface2_name=self.surface2_name
                )

                # Skip if any result file already exists
                if any(os.path.exists(path) for path in check_feature_paths):
                    print(f"Skipping '{feature}' -- result already exists.")
                    continue
                
                dist = compute_feature_distance(T, feature, self.surface1_name, self.surface2_name, top_k=2)
                save_results(dist, feature, self.surface1_name, self.surface2_name)


    
    def scalar_field_cost(self, surface1_name, surface2_name):
        surf1_path = os.path.join(BASEPATH, 'data', 'processed', surface1_name, f'{surface1_name}')
        surf2_path = os.path.join(BASEPATH, 'data', 'processed', surface2_name, f'{surface2_name}')
        feat1 = []
        feat2 = []
        # Load features for both surfaces
        for feature in self.features_list:
            # Load standard feature
            f1 = np.load(f'{surf1_path}_{feature}.npy')
            f2 = np.load(f'{surf2_path}_{feature}.npy')
            
            feat1.append(f1)
            feat2.append(f2)

            F1 = np.stack(feat1, axis=1)
            F2 = np.stack(feat2, axis=1)

            M = ot.dist(F1, F2, metric='seuclidean') ** 2  # Shape: (surface1.vertices.shape[0], surface2.vertices.shape[0])
        return M


