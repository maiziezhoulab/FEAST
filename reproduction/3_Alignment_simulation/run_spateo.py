import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
device = "cuda"
print("Running this notebook on: ", device)

import spateo as st
print("Last run with spateo version:", st.__version__)

# Other imports
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

def run_spateo(slice1, slice2):
    # preprocess slice1
    sc.pp.filter_cells(slice1, min_genes=1)  # we use min_genes=10 as 100 is too large for ST data
    sc.pp.filter_genes(slice1, min_cells=1)
    # Saving count data
    slice1.layers["counts"] = slice1.X.copy()
    # Normalizing to median total counts
    sc.pp.normalize_total(slice1)
    # Logarithmize the data
    sc.pp.log1p(slice1)

    # preprocess slice1
    sc.pp.filter_cells(slice2, min_genes=1)
    sc.pp.filter_genes(slice2, min_cells=1)
    # Saving count data
    slice2.layers["counts"] = slice2.X.copy()
    # Normalizing to median total counts
    sc.pp.normalize_total(slice2)
    # Logarithmize the data
    sc.pp.log1p(slice2)

    st.align.group_pca([slice1,slice2], pca_key='X_pca')

    skey_added = 'spatial_aligned'
    # spateo return aligned slices as well as the mapping matrix
    aligned_slices, pis = st.align.morpho_align(
    ## Uncomment this if use highly variable genes
    models=[slice1, slice2],
    ## Uncomment the following if use pca embeddings
    rep_layer='X_pca',
    rep_field='obsm',
    dissimilarity='cos',
        verbose=False,
        spatial_key="spatial",
        key_added=skey_added,
        device=device,
        return_mapping=True
        )
    print(pis[0].shape)
    return aligned_slices, pis[0]

def convert_int_keys_to_str(adata):
    """Convert integer keys in the uns dictionary to strings to avoid h5ad saving errors."""
    # Process main uns dictionary
    for key in list(adata.uns.keys()):
        if isinstance(adata.uns[key], dict):
            # Recursively process nested dictionaries
            _fix_nested_dict(adata.uns[key])
    return adata

def _fix_nested_dict(d):
    """Recursively convert integer keys to strings in nested dictionaries."""
    for key in list(d.keys()):
        # Convert int keys to strings
        if isinstance(key, int):
            d[str(key)] = d.pop(key)
        
        # Process nested dictionaries
        if isinstance(d[key if isinstance(key, str) else str(key)], dict):
            _fix_nested_dict(d[key if isinstance(key, str) else str(key)])
    return d

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice1", type=str, required=True)
    parser.add_argument("--slice2", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    slice1 = sc.read_h5ad(args.slice1)
    slice2 = sc.read_h5ad(args.slice2)
    aligned_slices, pis = run_spateo(slice1, slice2)
    
    # Fix integer keys in uns dictionaries before saving
    aligned_slices[0] = convert_int_keys_to_str(aligned_slices[0])
    aligned_slices[1] = convert_int_keys_to_str(aligned_slices[1])
    
    aligned_slices[0].write_h5ad(os.path.join(args.output_dir, f"{args.slice1.split('/')[-1].split('.')[0]}_paste_aligned.h5ad"))
    aligned_slices[1].write_h5ad(os.path.join(args.output_dir, f"{args.slice2.split('/')[-1].split('.')[0]}_paste_aligned.h5ad"))
    np.save(os.path.join(args.output_dir, f"{args.slice1.split('/')[-1].split('.')[0]}_paste_pis.npy"), pis)
