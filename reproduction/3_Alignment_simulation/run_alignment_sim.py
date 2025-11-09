"""
Alignment Simulation: Rotation-based transformation benchmark.

Usage:
    python run_alignment_sim.py --input <dataset> --output-dir <dir>
"""

import sys
import os
import argparse
from pathlib import Path
import scanpy as sc
from FEAST import STmulator

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_paths import DATASETS, check_dataset_exists


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run alignment simulation with rotation transformations')
    parser.add_argument('--input', type=str, default='align_dlpfc',
                        help='Dataset shortcut or path (default: align_dlpfc)')
    parser.add_argument('--output-dir', type=str, default='alignment_sim_output',
                        help='Output directory for simulated files')
    args = parser.parse_args()
    
    # Resolve input path
    if args.input in DATASETS:
        input_path = DATASETS[args.input]
        print(f"Using dataset shortcut '{args.input}' → {input_path}")
        check_dataset_exists(input_path)
    else:
        input_path = Path(args.input)
        check_dataset_exists(input_path)
    
    adata = sc.read_h5ad(str(input_path))
    adata = sc.read_h5ad(str(input_path))
    sc.pp.filter_genes(adata, min_cells=30)
    adata = adata[~adata.obs['sce.layer_guess'].isna(), :]
    print(f"Filtered adata: {adata}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    simulator = STmulator(adata)

    simulator = STmulator(adata)

    for angle in [1, 5, 10, 30, 45, 60]:
        original_adata, transformed_adata = simulator.simulate_alignment(
            transformation_type='rotation',
            rotation_angle=angle,
            data_type='sequencing',
            sigma=0,
            warp_strength=0
        )

        if angle == 1:
            original_adata.write_h5ad(output_dir / "original_adata.h5ad")
            print(f"Saved original adata to: {output_dir / 'original_adata.h5ad'}")

        transformed_adata.write_h5ad(output_dir / f"transformed_adata_angle_{angle}.h5ad")
        print(f"Saved transformed adata for angle {angle}")
    
    print(f"\nAlignment simulation complete! Results saved to: {output_dir}")