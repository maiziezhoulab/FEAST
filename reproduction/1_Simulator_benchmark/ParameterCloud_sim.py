"""
Simulator Benchmark: Single-slice simulation using FEAST simulator.

Usage:
    python ParameterCloud_sim.py --input <dataset> --output <path>
    
Example:
    python ParameterCloud_sim.py --input sim_dlpfc --output results/simulated.h5ad
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from FEAST import simulator

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_paths import DATASETS, get_data_path, check_dataset_exists


def single_slice_sim(adata_path, output_path):
    # Load the AnnData object
    adata = sc.read_h5ad(adata_path)

    sc.pp.filter_genes(adata, min_cells=30)
    common_genes = adata.var_names.tolist()
    exp_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.copy()
    stats = pd.DataFrame(
        {'mean': np.mean(exp_matrix, axis=0),
            'variance': np.var(exp_matrix, axis=0),
            'zero_prop': 1 - (np.count_nonzero(exp_matrix, axis=0) / adata.n_obs)},
        index=common_genes
    ).clip(lower=1e-10)

    simulated_adata = simulator.simulate_single_slice(
        adata=adata,
        visualize_fits=True, 
        use_real_stats_directly=True,
        use_heuristic_search=True,
        follower_sigma_factor=0,
        sigma=0,
        min_accepted_error=0.0001,
        screening_pool_size=50000,
        assignment_weights = {'mean': 1, 'variance': 1, 'zero_prop': 1.0},
        top_n_to_fully_evaluate=10
        )   
    
    simulated_adata.write_h5ad(output_path)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(
        description='Run single-slice simulation benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined dataset shortcut
  python ParameterCloud_sim.py --input sim_dlpfc --output results/sim_dlpfc.h5ad
  
  # Use custom dataset file
  python ParameterCloud_sim.py --input /path/to/data.h5ad --output results/simulated.h5ad
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Dataset shortcut (e.g., 'sim_dlpfc') or path to input AnnData file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output simulated AnnData file")
    args = parser.parse_args()
    
    # Resolve input path
    if args.input in DATASETS:
        input_path = DATASETS[args.input]
        print(f"Using dataset shortcut '{args.input}' → {input_path}")
        check_dataset_exists(input_path)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
    
    single_slice_sim(str(input_path), args.output)

