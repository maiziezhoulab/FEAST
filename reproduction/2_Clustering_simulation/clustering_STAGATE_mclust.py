import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import anndata as ad
import scipy.sparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf
    if hasattr(tf, 'compat'):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print(f"TensorFlow setup warning: {e}")


def run_stagate_mclust(adata, num_cluster=7, save_df_path=None):
    adata.layers["counts"] = adata.X.copy()
    adata.var_names_make_unique()
    print("preprocessed data for STAGATE clustering...")
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    import STAGATE
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAGATE.Stats_Spatial_Net(adata)
    # generate random seed
    random_seed = random.randint(0, 10000)

    adata = STAGATE.train_STAGATE(adata, alpha=0, random_seed=random_seed)
    
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=num_cluster, random_seed=random_seed)
    
    # Save the clustering results to a new DataFrame.
    if save_df_path:
        # Create a new DataFrame with the index from adata.obs
        df = pd.DataFrame(index=adata.obs.index)
        # Add the clustering results
        df['cluster2_STAGATE'] = adata.obs['mclust'].astype(str)
        # Save to the specified path
        df.to_csv(save_df_path)
        
    adata.X = adata.layers["counts"]
    
    return adata


import argparse

def main():
    parser = argparse.ArgumentParser(description="STAGATE clustering script")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--num_cluster", type=int, default=7, help="Number of clusters")
    parser.add_argument("--save_df_path", type=str, help="Path to save the clustering results DataFrame")

    args = parser.parse_args()


    adata = sc.read(args.input)
    # covert X to sparse matrix
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adata = run_stagate_mclust(adata, args.num_cluster, args.save_df_path)

if __name__ == "__main__":
    main()