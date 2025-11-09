import pandas as pd
import scanpy as sc

def convert_parquet_to_anndata(adata, parquet_file):
    # Get clustering results (no longer including PCA clustering)
    clustering_results2 = adata.obs['cluster2_STAGATE']
    clustering_results3 = adata.obs['cluster3_graphst']
    
    # Convert to DataFrame
    results_df = pd.DataFrame({
        'cluster2_STAGATE': clustering_results2, 
        'cluster3_graphst': clustering_results3, 
    })
    results_df.index = adata.obs.index
    
    # Convert any NumPy types to native Python types
    results_df = results_df.astype(str)  # or use appropriate type conversion

    # Save to CSV
    results_df.to_csv(parquet_file)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert clustering results to parquet format.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input file (e.g., 'path/to/input.h5ad').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output file (e.g., 'path/to/output.parquet').",
    )
    args = parser.parse_args()
    adata = sc.read_h5ad(args.input_file)
    convert_parquet_to_anndata(adata, args.output_file)

if __name__ == "__main__":
    main()