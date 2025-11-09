import os
import torch
import pandas as pd
import scanpy as sc
from GraphST import GraphST
from GraphST.utils import clustering
import scipy as sp
import random
import numpy as np
# GraphST need full genes; and we do not use refinement parameter
def process_graphst_clustering(adata, n_clusters, radius=50, tool='mclust', refinement=False, 
                                start=0.1, end=2.0, increment=0.01, save_df_path=None):
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    
    # Check if CUDA is available and choose appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device for GraphST")
    else:
        device = "cpu"
        print("CUDA not available, using CPU device for GraphST")
    # generate a random seed
    random_seed = random.randint(0, 10000)
    model = GraphST.GraphST(adata, device=device, random_seed=random_seed)
    
    print("Training GraphST model...")
    adata = model.train()

    
    print(f"Performing clustering using {tool} method...")
    if tool == 'mclust':
        clustering(adata, n_clusters, radius=radius, method=tool, refinement=refinement)
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, radius=radius, method=tool, start=start, end=end, increment=increment, refinement=False)
    else:
        raise ValueError(f"Unsupported clustering tool: {tool}. Choose from 'mclust', 'leiden', or 'louvain'.")
    
    print(f"Clustering completed with method: {tool}.")
    
    # Save the clustering results to a new DataFrame.
    if save_df_path:
        # Create a new DataFrame with the index from adata.obs
        df = pd.DataFrame(index=adata.obs.index)
        # Add the clustering results
        df['cluster3_graphst'] = adata.obs['mclust'].astype(str)
        # Save to the specified path
        df.to_csv(save_df_path)
        
    adata.X = adata.layers["counts"]

    adata.X = sp.sparse.csr_matrix(adata.X)

    return adata


import argparse

def main():
    parser = argparse.ArgumentParser(description="GraphST clustering script")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--n_clusters", type=int, help="Number of clusters")
    parser.add_argument("--radius", type=int, default=50, help="Radius for clustering")
    parser.add_argument("--tool", type=str, default="mclust", help="Clustering tool")
    parser.add_argument("--refinement", type=bool, default=True, help="Refinement")
    parser.add_argument("--start", type=float, default=0.1, help="Start")
    parser.add_argument("--end", type=float, default=2.0, help="End")
    parser.add_argument("--increment", type=float, default=0.01, help="Increment")
    parser.add_argument("--save_df_path", type=str, help="Path to save the clustering results DataFrame")
    args = parser.parse_args()

    adata = sc.read(args.input)

    adata = process_graphst_clustering(adata, args.n_clusters, args.radius, args.tool, args.refinement, args.start, args.end, args.increment, args.save_df_path)
    
if __name__ == "__main__":
    main()