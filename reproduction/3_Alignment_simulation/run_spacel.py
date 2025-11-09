import SPACEL
from SPACEL import Scube
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import os
import argparse
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

def calculate_layer_match_pis(adata_list, layer_key='annotation', use_hungarian=True):
    """
    Create a one-to-one match matrix between spots in two aligned datasets.
    Each spot is matched to exactly one spot in the other dataset based on
    minimum distance after alignment.
    
    Parameters
    ----------
    adata_list : list of AnnData
        List containing two aligned AnnData objects
    layer_key : str or None
        Key for layer/cluster annotations if available
    use_hungarian : bool
        Whether to use Hungarian algorithm for optimal matching
        
    Returns
    -------
    match_matrix : ndarray
        Binary matrix indicating matches between spots
    """
    # Validate input
    if len(adata_list) != 2:
        raise ValueError("adata_list must contain exactly 2 AnnData objects")
    
    # Extract spatial coordinates from both aligned datasets
    coords1, coords2 = None, None
    
    # Try aligned coordinates first, then fall back to original spatial
    for coords_key in ['spatial_aligned', 'spatial']:
        if coords_key in adata_list[0].obsm and coords_key in adata_list[1].obsm:
            coords1 = np.array(adata_list[0].obsm[coords_key])
            coords2 = np.array(adata_list[1].obsm[coords_key])
            print(f"Using coordinates from '{coords_key}'")
            break
    
    if coords1 is None or coords2 is None:
        raise ValueError("No spatial coordinates found in either 'spatial_aligned' or 'spatial'")
    
    # Validate coordinate dimensions
    if coords1.shape[1] != coords2.shape[1]:
        raise ValueError(f"Coordinate dimensions don't match: {coords1.shape[1]} vs {coords2.shape[1]}")
    
    # Extract layer information if available
    use_layer = False
    layer_info1, layer_info2 = None, None
    
    if (layer_key is not None and 
        layer_key in adata_list[0].obs and 
        layer_key in adata_list[1].obs):
        layer_info1 = adata_list[0].obs[layer_key].values
        layer_info2 = adata_list[1].obs[layer_key].values
        use_layer = True
        print(f"Using layer information from '{layer_key}'")
    else:
        print("No layer information available, matching based on distance only")
    
    n_spots1, n_spots2 = len(adata_list[0]), len(adata_list[1])
    
    if use_hungarian and not use_layer:
        # Use efficient Hungarian algorithm for optimal matching without layer constraints
        return _hungarian_matching(coords1, coords2)
    
    elif use_layer:
        # Use layer-constrained matching
        return _layer_constrained_matching(coords1, coords2, layer_info1, layer_info2, use_hungarian)
    
    else:
        # Use efficient nearest neighbor matching
        return _nearest_neighbor_matching(coords1, coords2)


def _hungarian_matching(coords1, coords2):
    """Optimal matching using Hungarian algorithm"""
    from scipy.spatial.distance import cdist
    
    # Calculate all pairwise distances
    distance_matrix = cdist(coords1, coords2)
    
    # Use Hungarian algorithm for optimal assignment
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    
    # Create match matrix
    match_matrix = np.zeros((len(coords1), len(coords2)))
    match_matrix[row_indices, col_indices] = 1
    
    avg_distance = distance_matrix[row_indices, col_indices].mean()
    print(f"Hungarian matching completed. Average distance: {avg_distance:.4f}")
    
    return match_matrix


def _nearest_neighbor_matching(coords1, coords2):
    """Efficient nearest neighbor matching using KDTree"""
    # Build KDTree for efficient nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords2)
    distances, indices = nbrs.kneighbors(coords1)
    
    # Create match matrix
    match_matrix = np.zeros((len(coords1), len(coords2)))
    
    # Handle potential duplicate matches
    used_indices = set()
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if idx not in used_indices:
            match_matrix[i, idx] = 1
            used_indices.add(idx)
        else:
            # Find next best available match
            all_distances = np.linalg.norm(coords1[i] - coords2, axis=1)
            sorted_indices = np.argsort(all_distances)
            for next_idx in sorted_indices:
                if next_idx not in used_indices:
                    match_matrix[i, next_idx] = 1
                    used_indices.add(next_idx)
                    break
    
    avg_distance = distances.mean()
    print(f"Nearest neighbor matching completed. Average distance: {avg_distance:.4f}")
    
    return match_matrix


def _layer_constrained_matching(coords1, coords2, layer_info1, layer_info2, use_hungarian=True):
    """Matching with layer/annotation constraints"""
    n_spots1, n_spots2 = len(coords1), len(coords2)
    match_matrix = np.zeros((n_spots1, n_spots2))
    
    # Get unique layers
    unique_layers = np.unique(np.concatenate([layer_info1, layer_info2]))
    print(f"Found {len(unique_layers)} unique layers: {unique_layers}")
    
    total_matches = 0
    for layer in unique_layers:
        # Find spots in current layer
        layer1_mask = layer_info1 == layer
        layer2_mask = layer_info2 == layer
        
        layer1_indices = np.where(layer1_mask)[0]
        layer2_indices = np.where(layer2_mask)[0]
        
        if len(layer1_indices) == 0 or len(layer2_indices) == 0:
            print(f"Skipping layer '{layer}' - no spots in one or both datasets")
            continue
        
        layer1_coords = coords1[layer1_mask]
        layer2_coords = coords2[layer2_mask]
        
        # Match within this layer
        if use_hungarian and len(layer1_coords) > 0 and len(layer2_coords) > 0:
            layer_matches = _hungarian_matching(layer1_coords, layer2_coords)
        else:
            layer_matches = _nearest_neighbor_matching(layer1_coords, layer2_coords)
        
        # Map back to original indices
        for i, layer1_idx in enumerate(layer1_indices):
            for j, layer2_idx in enumerate(layer2_indices):
                if layer_matches[i, j] == 1:
                    match_matrix[layer1_idx, layer2_idx] = 1
                    total_matches += 1
        
        print(f"Layer '{layer}': {np.sum(layer_matches)} matches from {len(layer1_coords)} vs {len(layer2_coords)} spots")
    
    print(f"Total matches across all layers: {total_matches}")
    return match_matrix

def run_spacel(adata1, adata2, layer_key='annotation', use_hungarian=True):
    """Run the Spacel alignment algorithm.
    
    Parameters
    ----------
    adata1 : AnnData
        First slice (reference)
    adata2 : AnnData
        Second slice (to be aligned)
    layer_key : str or None
        Key for observation/cluster annotations if available
    use_hungarian : bool
        Whether to use Hungarian algorithm for optimal matching
        
    Returns
    -------
    aligned_slices : list of AnnData
        Aligned slices
    pis : ndarray
        Alignment matrix
    """
    print(f"Starting SPACEL alignment with {len(adata1)} and {len(adata2)} spots")
    
    # Validate input data
    if len(adata1) == 0 or len(adata2) == 0:
        raise ValueError("Input datasets cannot be empty")
    
    # Handle layer_key cases with better validation
    effective_layer_key = None
    if layer_key is not None:
        # Try the provided layer key first
        if layer_key in adata1.obs.columns and layer_key in adata2.obs.columns:
            effective_layer_key = layer_key
            adata1.obs[layer_key] = adata1.obs[layer_key].astype('str')
            adata2.obs[layer_key] = adata2.obs[layer_key].astype('str')
            print(f"Using layer key '{layer_key}' for alignment")
        # Try alternative layer keys
        elif 'sce.layer_guess' in adata1.obs.columns and 'sce.layer_guess' in adata2.obs.columns:
            effective_layer_key = 'sce.layer_guess'
            adata1.obs['sce.layer_guess'] = adata1.obs['sce.layer_guess'].astype('str')
            adata2.obs['sce.layer_guess'] = adata2.obs['sce.layer_guess'].astype('str')
            print(f"Using found layer key 'sce.layer_guess' for alignment")
        else:
            print(f"Warning: Layer key '{layer_key}' not found in one or both datasets")
            print("Available columns in adata1:", list(adata1.obs.columns))
            print("Available columns in adata2:", list(adata2.obs.columns))
            effective_layer_key = None
    
    # Create copy to avoid modifying original data
    adata_list = [adata1.copy(), adata2.copy()]
    
    try:
        # Run SPACEL alignment
        print("Running SPACEL Scube.align...")
        
        # Call SPACEL with or without cluster key
        if effective_layer_key is not None:
            print(f"Running SPACEL with cluster key: {effective_layer_key}")
            Scube.align(adata_list,
                cluster_key=effective_layer_key,
                n_neighbors=15,
                n_threads=12,
            )
        else:
            print("Running SPACEL without cluster information")
            Scube.align(adata_list,
                n_neighbors=15,
                n_threads=12,
            )
        
        print("SPACEL alignment completed successfully")
        
        # Calculate matching matrix
        print("Calculating spot matches...")
        pis = calculate_layer_match_pis(adata_list, effective_layer_key, use_hungarian)
        
        # Validate results
        n_matches = np.sum(pis)
        print(f"Created {n_matches} spot matches out of {len(adata1)} possible")
        
        if n_matches == 0:
            print("Warning: No matches found between datasets!")
        
        return adata_list, pis
        
    except Exception as e:
        print(f"Error during SPACEL alignment: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPACEL alignment on two spatial transcriptomics slices")
    parser.add_argument("--slice1", type=str, required=True, help="Path to first slice H5AD file")
    parser.add_argument("--slice2", type=str, required=True, help="Path to second slice H5AD file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--layer_key", type=str, default="annotation", help="Layer/cluster annotation key in obs")
    parser.add_argument("--use_hungarian", action="store_true", default=True, help="Use Hungarian algorithm for optimal matching")
    parser.add_argument("--no_hungarian", action="store_true", help="Disable Hungarian algorithm")
    args = parser.parse_args()
    
    # Handle hungarian argument
    use_hungarian = args.use_hungarian and not args.no_hungarian
    
    # Validate input files
    if not os.path.exists(args.slice1):
        raise FileNotFoundError(f"Slice1 file not found: {args.slice1}")
    if not os.path.exists(args.slice2):
        raise FileNotFoundError(f"Slice2 file not found: {args.slice2}")
    
    print(f"Loading slice1 from {args.slice1}")
    try:
        slice1 = sc.read_h5ad(args.slice1)
        print(f"Loaded slice1: {slice1.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load slice1: {e}")
    
    print(f"Loading slice2 from {args.slice2}")
    try:
        slice2 = sc.read_h5ad(args.slice2)
        print(f"Loaded slice2: {slice2.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load slice2: {e}")
    
    print("Running Spacel alignment...")
    aligned_slices, pis = run_spacel(slice1, slice2, args.layer_key, use_hungarian)
    
    # Determine output paths
    if args.output_dir:
        # Use provided output directory
        os.makedirs(args.output_dir, exist_ok=True)
        slice1_output = os.path.join(args.output_dir, "slice1_spacel_aligned.h5ad")
        slice2_output = os.path.join(args.output_dir, "slice2_spacel_aligned.h5ad")
        pi_output = os.path.join(args.output_dir, "spacel_pis.npy")
    else:
        # Default to original file path pattern
        slice1_output = args.slice1.replace(".h5ad", "_spacel_aligned.h5ad")
        slice2_output = args.slice2.replace(".h5ad", "_spacel_aligned.h5ad") 
        pi_output = args.slice1.replace(".h5ad", "_spacel_pis.npy")
    
    print(f"Saving aligned slice1 to {slice1_output}")
    try:
        aligned_slices[0].write_h5ad(slice1_output)
    except Exception as e:
        print(f"Warning: Failed to save slice1: {e}")
    
    print(f"Saving aligned slice2 to {slice2_output}")
    try:
        aligned_slices[1].write_h5ad(slice2_output)
    except Exception as e:
        print(f"Warning: Failed to save slice2: {e}")
    
    print(f"Saving alignment matrix to {pi_output}")
    try:
        np.save(pi_output, pis)
        print(f"Alignment matrix shape: {pis.shape}, non-zero entries: {np.count_nonzero(pis)}")
    except Exception as e:
        print(f"Warning: Failed to save alignment matrix: {e}")
    
    print("Spacel alignment completed successfully!")