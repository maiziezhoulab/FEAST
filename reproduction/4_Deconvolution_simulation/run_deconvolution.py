"""
Deconvolution Simulation: Multi-resolution cell type deconvolution benchmark.

Usage:
    python run_deconvolution.py --input <dataset> --output-dir <dir>
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from FEAST.deconvolution import DeconvolutionSimulator, simulate_deconvolution_from_single_cells

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_paths import DATASETS, check_dataset_exists


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run deconvolution simulation at multiple resolutions')
    parser.add_argument('--input', type=str, default='deconv_merfish',
                        help='Dataset shortcut or path (default: deconv_merfish)')
    parser.add_argument('--output-dir', type=str, default='deconvolution_sim_output',
                        help='Output directory for results')
    parser.add_argument('--cell-type-key', type=str, default='CellType',
                        help='Cell type annotation key in adata.obs')
    args = parser.parse_args()
    
    # Resolve input path
    if args.input in DATASETS:
        input_path = DATASETS[args.input]
        print(f"Using dataset shortcut '{args.input}' → {input_path}")
        check_dataset_exists(input_path)
    else:
        input_path = Path(args.input)
        check_dataset_exists(input_path)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(input_path))

    adata = sc.read_h5ad(str(input_path))

    print("=== Deconvolution Simulation with Ground Truth ===")
    print(f"Original data: {adata.shape}")
    print(f"Available obsm keys: {list(adata.obsm.keys())}")
    print(f"Available obs columns: {list(adata.obs.columns)}")

    adata.var_names_make_unique()
    print(f"Processed data: {adata}")

    cell_type_key = args.cell_type_key
    if cell_type_key not in adata.obs.columns:
        possible_keys = ['cell_type', 'Cell_Type', 'celltype', 'cluster', 'annotation']
        for key in possible_keys:
            if key in adata.obs.columns:
                cell_type_key = key
                break
        else:
            cell_type_key = None

    if cell_type_key:
        print(f"\nUsing cell type key: {cell_type_key}")
        print(f"Cell types found: {adata.obs[cell_type_key].value_counts()}")
        print(f"Total cell types: {adata.obs[cell_type_key].nunique()}")
    else:
        print("No cell type annotations found. Will simulate without ground truth.")

    print("\n" + "="*60)
    for resolution in [0.1, 0.2, 0.25, 0.5]:
        print(f"\n>>> PROCESSING RESOLUTION: {resolution} <<<")
        
        deconv_data = simulate_deconvolution_from_single_cells(
            reference_adata=adata,
            cell_type_key=cell_type_key,
            downsampling_factor=resolution,
            grid_type='hexagonal',
            sigma=0.01,
            alpha=0.005,
            verbose=True
        )
        
        print(f"\n--- Results for Resolution {resolution} ---")
        print(f"Deconvolution data shape: {deconv_data.shape}")
        print(f"Available obsm keys: {list(deconv_data.obsm.keys())}")
        print(f"Available uns keys: {list(deconv_data.uns.keys())}")
        
        if 'cell_type_proportions' in deconv_data.obsm:
            proportions = deconv_data.obsm['cell_type_proportions']
            cell_type_names = deconv_data.uns['cell_type_names']
            print(f"Ground truth proportions shape: {proportions.shape}")
            print(f"Cell types: {cell_type_names}")
            print(f"Mean proportions per cell type:")
            for i, ct in enumerate(cell_type_names):
                mean_prop = proportions[:, i].mean()
                print(f"  {ct}: {mean_prop:.3f}")
        
        output_file = output_dir / f"deconv_simulation_resolution_{resolution}.h5ad"
        deconv_data.write_h5ad(str(output_file))
        print(f"✓ Saved to: {output_file}")

    print(f"\n{'='*60}")
    print(f"Deconvolution simulation complete! Results saved to: {output_dir}")
    print(f"{'='*60}\n")