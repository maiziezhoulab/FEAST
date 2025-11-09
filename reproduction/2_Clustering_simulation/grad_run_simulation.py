"""
Clustering Simulation: Marginal alteration experiments.

Usage:
    python grad_run_simulation.py --input <dataset> --output-dir <dir>
"""

import sys
import os
import argparse
from pathlib import Path
import scanpy as sc
from FEAST import simulator
from FEAST.modeling.marginal_alteration import AlterationConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_paths import DATASETS, check_dataset_exists


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run clustering simulation with marginal alterations')
    parser.add_argument('--input', type=str, default='clust_dlpfc',
                        help='Dataset shortcut or path (default: clust_dlpfc)')
    parser.add_argument('--output-dir', type=str, default='grad_sim_output',
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
    
    h5ad_file = sc.read(str(input_path))
    h5ad_file = sc.read(str(input_path))
    sc.pp.filter_cells(h5ad_file, min_genes=40)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_adata = h5ad_file.copy()
    input_adata.write_h5ad(output_dir / "raw_adata.h5ad")
    
    print("Running baseline simulation (no alterations)...")
    no_alter_simulation = simulator.simulate_single_slice(input_adata, min_accepted_error=0.0001,
            screening_pool_size=50000)
    no_alter_simulation.write_h5ad(output_dir / "no_alter_simulation.h5ad")

    print("\nRunning controlled experiment 1: Mean-only alterations...")
    for mean_fold_change in [ 0.9, 0.95,0.99,1, 1.05, 1.1, 1.2]:  
        print(f"  Testing mean fold change: {mean_fold_change}x")
        adata = input_adata.copy()
        
        # Create mean-only alteration config
        alteration_config = AlterationConfig.mean_only(
            fold_change=mean_fold_change,
            strength=0.3
        )
        
        simulated_adata = simulator.simulate_single_slice(
            adata, 
            alteration_config=alteration_config,
            verbose=False,
            use_real_stats_directly=False,
            use_heuristic_search=True,
            follower_sigma_factor=0,
            sigma=0,
            min_accepted_error=0.0001,
            screening_pool_size=10000,
            assignment_weights = {'mean': 1, 'variance': 1, 'zero_prop': 1.0},
            top_n_to_fully_evaluate=10
        )
        simulated_adata.write_h5ad(output_dir / f"mean_only_fc_{mean_fold_change}.h5ad")

    # Controlled Experiment 2: Variance-only alterations
    print("\nRunning controlled experiment 2: Variance-only alterations...")
    for var_fold_change in [1.01, 1.05, 0.99,0.95]:  
        print(f"  Testing variance fold change: {var_fold_change}x")
        adata = input_adata.copy()
        
        # Create variance-only alteration config
        alteration_config = AlterationConfig.variance_only(
            fold_change=var_fold_change,
            strength=0.3
        )
        
        simulated_adata = simulator.simulate_single_slice(
            adata,    
            alteration_config=alteration_config,
            verbose=False,
            use_real_stats_directly=False,
            use_heuristic_search=True,
            follower_sigma_factor=0,
            sigma=0,
            min_accepted_error=0.0001,
            screening_pool_size=10000,
            assignment_weights = {'mean': 1, 'variance': 1, 'zero_prop': 1.0},
            top_n_to_fully_evaluate=10
        )
        simulated_adata.write_h5ad(output_dir / f"variance_only_fc_{var_fold_change}.h5ad")

    print("\nRunning controlled experiment 3: Sparsity-only alterations...")
    for sparsity_fold_change in [0.9 ,0.95, 0.99,1.0, 1.01,1.05, 1.1, 1.2]:  
        print(f"  Testing sparsity fold change: {sparsity_fold_change}x")
        adata = input_adata.copy()
        
        # Create sparsity-only alteration config
        alteration_config = AlterationConfig.sparsity_only(
            fold_change=sparsity_fold_change,
            strength=0.3
        )
        
        simulated_adata = simulator.simulate_single_slice(
           adata, 
            alteration_config=alteration_config,
            verbose=False,
            use_real_stats_directly=False,
            use_heuristic_search=True,
            follower_sigma_factor=0,
            sigma=0,
            min_accepted_error=0.0001,
            screening_pool_size=10000,
            assignment_weights = {'mean': 1, 'variance': 1, 'zero_prop': 1.0},
            top_n_to_fully_evaluate=10
        )
        simulated_adata.write_h5ad(output_dir / f"sparsity_only_fc_{sparsity_fold_change}.h5ad")

    print(f"\nAll controlled experiments complete! Results saved to: {output_dir}")
    print("Experiment summary:")
    print("  - Baseline: no_alter_simulation.h5ad")
    print("  - Mean-only: mean_only_fc_*.h5ad (8 conditions)")
    print("  - Variance-only: variance_only_fc_*.h5ad (8 conditions)")
    print("  - Sparsity-only: sparsity_only_fc_*.h5ad (9 conditions)")
    print("  - Total: 26 simulation files")


        