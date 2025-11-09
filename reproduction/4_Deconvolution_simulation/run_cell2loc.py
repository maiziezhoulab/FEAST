import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import cell2location
from cell2location.models import RegressionModel
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text for PDFs
import scipy.sparse as sparse
import torch # Keep torch for checking availability
import pandas as pd
import gc
import argparse
# Determine accelerator and devices for scvi-tools/PyTorch Lightning
if torch.cuda.is_available():
    accelerator = "cuda"
    devices = 1 # Use 1 GPU (typically the first one, ID 0)
    # If you want to specify a particular GPU or multiple, e.g., [0] or [0, 1]
    # devices = [0] # To use GPU 0
else:
    accelerator = "cpu"
    devices = "auto" # or 1 for one CPU core, "auto" is usually fine

def run_cell2loc(adata_ref, adata_decon, cell_type_key='CellType'):
    # prepare anndata for the regression model
    print(f"Using cell type key: {cell_type_key}")
    cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
    labels_key=cell_type_key,
    )

    mod = RegressionModel(adata_ref)

    # Pass accelerator and devices arguments
    mod.train(accelerator=accelerator) # Default max_epochs is 250

    adata_ref = mod.export_posterior(
    adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
    )

    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    if sparse.issparse(adata_decon.X):
        # Ensure adata_ref.X is also sparse if adata_decon.X is sparse, and convert to CSR
        if not sparse.issparse(adata_ref.X) or not sparse.isspmatrix_csr(adata_ref.X):
             adata_ref.X = sparse.csr_matrix(adata_ref.X) # Convert to sparse CSR
        if not sparse.isspmatrix_csr(adata_decon.X):
            adata_decon.X = adata_decon.X.tocsr()
    else: # If adata_decon.X is dense, ensure adata_ref.X is also treated as dense then converted
        adata_ref.X = sparse.csr_matrix(adata_ref.X) # Convert to sparse CSR, assumes it can be dense or sparse
        adata_decon.X = sparse.csr_matrix(adata_decon.X) # Convert to sparse CSR

    common_genes = adata_decon.var_names.intersection(inf_aver.index)
    if len(common_genes) == 0:
        raise ValueError("No common genes found between reference signatures (inf_aver) "
                         "and spatial data (adata_decon). Ensure gene names are consistent.")
    print(f"Found {len(common_genes)} common genes. "
          f"Original genes in reference: {len(inf_aver.index)}, "
          f"Original genes in spatial data: {len(adata_decon.var_names)}.")
    adata_decon = adata_decon[:, common_genes].copy()
    inf_aver = inf_aver.loc[common_genes, :].copy()

    cell2location.models.Cell2location.setup_anndata(adata=adata_decon)
    mod = cell2location.models.Cell2location(
    adata_decon, cell_state_df=inf_aver,
    N_cells_per_location=10,
    detection_alpha=20
    )

    # Pass accelerator and devices arguments
    mod.train(max_epochs=20000, accelerator=accelerator)

    adata_decon = mod.export_posterior(
    adata_decon, sample_kwargs={'num_samples': 1000, 'batch_size': adata_decon.shape[0]}
    )
    if 'q05_cell_abundance_w_sf' not in adata_decon.obsm:
        raise KeyError("Key 'q05_cell_abundance_w_sf' not found in adata_decon.obsm. ")
    if 'mod' not in adata_decon.uns or 'factor_names' not in adata_decon.uns['mod']:
        raise KeyError("Key 'factor_names' not found in adata_decon.uns['mod']. ")
    adata_decon.obs[adata_decon.uns['mod']['factor_names']] = adata_decon.obsm['q05_cell_abundance_w_sf']
    return get_cell_abundance_from_obs(adata_decon)

def get_cell_abundance_from_obs(adata_decon):
    potential_metadata_cols = ['n_counts', '_indices', '_scvi_batch', '_scvi_labels', 'in_tissue', 'array_row', 'array_col', 'n_genes_by_counts', 'total_counts']

    if 'mod' in adata_decon.uns and 'factor_names' in adata_decon.uns['mod']:
        cell_type_cols = list(adata_decon.uns['mod']['factor_names'])
        cell_type_cols = [col for col in cell_type_cols if col in adata_decon.obs.columns]
    else:
        cell_type_cols = [col for col in adata_decon.obs.columns if col not in potential_metadata_cols and not col.startswith('_')]

    if not cell_type_cols:
        print("Warning: No cell type abundance columns identified in `adata_decon.obs`. ")
        return pd.DataFrame(index=adata_decon.obs.index)
    abundance_df = adata_decon.obs[cell_type_cols].copy()

    # clean up cuda memory
    torch.cuda.empty_cache()
    gc.collect()
    return abundance_df

if __name__ == "__main__":
    # Configuration
    reference_data_path = '/maiziezhou_lab6/chen_yr/exper_data/Merfish_Zhuang-ABCA-3_007.h5ad'
    decon_results_dir = '/maiziezhou_lab6/chen_yr/scripts/2Reproduce/Figure3_deconvolution/result'
    output_dir = '/maiziezhou_lab6/chen_yr/scripts/2Reproduce/Figure3_deconvolution/cell2loc_results'
    cell_type_key = 'CellType'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Running Cell2Location on Deconvolution Simulations ===")
    print(f"Reference data: {reference_data_path}")
    print(f"Deconvolution results: {decon_results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load reference data once
    print(f"\nLoading reference data: {reference_data_path}")
    adata_ref = sc.read_h5ad(reference_data_path)
    
    if not adata_ref.var_names.is_unique:
        print("Making reference gene names unique...")
        adata_ref.var_names_make_unique()
    
    # Check if cell type key exists
    if cell_type_key not in adata_ref.obs.columns:
        print(f"Warning: {cell_type_key} not found in reference data.")
        print(f"Available columns: {list(adata_ref.obs.columns)}")
        # Try to find a suitable cell type column
        for col in adata_ref.obs.columns:
            if 'type' in col.lower() or 'class' in col.lower():
                cell_type_key = col
                print(f"Using {col} instead")
                break
    
    print(f"Reference data shape: {adata_ref.shape}")
    print(f"Using cell type key: {cell_type_key}")
    print(f"Found {adata_ref.obs[cell_type_key].nunique()} cell types")
    
    # Find all deconvolution simulation files
    import glob
    decon_files = glob.glob(os.path.join(decon_results_dir, "deconv_simulation_resolution_*.h5ad"))
    decon_files.sort()
    
    if not decon_files:
        print(f"No deconvolution simulation files found in {decon_results_dir}")
        exit(1)
    
    print(f"\nFound {len(decon_files)} deconvolution simulation files:")
    for file in decon_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    success_count = 0
    results_summary = []
    
    for decon_file in decon_files:
        # Extract resolution from filename
        basename = os.path.basename(decon_file)
        resolution = basename.replace('deconv_simulation_resolution_', '').replace('.h5ad', '')
        
        print(f"\n{'='*60}")
        print(f"Processing resolution {resolution}")
        print(f"Input file: {basename}")
        
        try:
            # Load deconvolution data
            adata_decon = sc.read_h5ad(decon_file)
            
            if not adata_decon.var_names.is_unique:
                adata_decon.var_names_make_unique()
            
            print(f"Deconvolution data shape: {adata_decon.shape}")
            
            # Run cell2location
            print("Running cell2location...")
            df = run_cell2loc(adata_ref.copy(), adata_decon, cell_type_key)
            
            # Save results
            output_file = os.path.join(output_dir, f"cell2loc_results_resolution_{resolution}.csv")
            df.to_csv(output_file)
            
            print(f"✓ Results saved to: {os.path.basename(output_file)}")
            print(f"Result shape: {df.shape}")
            
            # Store summary info
            results_summary.append({
                'resolution': resolution,
                'input_file': basename,
                'output_file': os.path.basename(output_file),
                'n_spots': df.shape[0],
                'n_cell_types': df.shape[1],
                'status': 'success'
            })
            
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error processing resolution {resolution}: {e}")
            import traceback
            traceback.print_exc()
            
            results_summary.append({
                'resolution': resolution,
                'input_file': basename,
                'output_file': 'failed',
                'n_spots': 0,
                'n_cell_types': 0,
                'status': f'error: {str(e)}'
            })
    
    # Print final summary
    print(f"\n{'='*60}")
    print("CELL2LOCATION PROCESSING COMPLETE!")
    print(f"Successfully processed: {success_count}/{len(decon_files)} files")
    print(f"Results saved in: {output_dir}")
    
    # Save summary table
    summary_df = pd.DataFrame(results_summary)
    summary_file = os.path.join(output_dir, "processing_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Processing summary saved to: {os.path.basename(summary_file)}")
    
    print("\nResults summary:")
    for _, row in summary_df.iterrows():
        if row['status'] == 'success':
            print(f"  ✓ Resolution {row['resolution']}: {row['n_spots']} spots, {row['n_cell_types']} cell types")
        else:
            print(f"  ✗ Resolution {row['resolution']}: {row['status']}")
    
    if success_count == len(decon_files):
        print("\nAll files processed successfully! 🎉")
