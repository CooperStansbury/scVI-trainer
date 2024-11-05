import os
import sys
import shutil
import pandas as pd
import numpy as np
import scvi
import scanpy as sc
import anndata as an
import scanpy.external as sce
import scipy
import time
import sklearn
import torch
from scib_metrics.benchmark import Benchmarker
from sklearn.metrics import silhouette_score

sc.settings.verbosity = 3 


if __name__ == "__main__":
    torch.cuda.empty_cache()
    adata_path = sys.argv[1]
    output_path = sys.argv[2]

    # load input data
    adata = sc.read_h5ad(adata_path)
    adata.X = adata.layers['counts'].copy() # TODO: don't hardcode thie
    print(f"\n------------ raw data ------------")
    print(adata)
    
    sc.logging.print_memory_usage()

    print(f"--- BENCHMARKING ---")
    torch.cuda.empty_cache()
    
    bm = Benchmarker(
        adata,
        batch_key="dataset",
        label_key="cell_label",
        embedding_obsm_keys=[
            'X_pca',
            'X_scanorama',
            'X_harmony',
            'X_scvi', 
            'X_scanvi',
        ],
    )
    
    bm.benchmark()

    # store the results
    bmdf = bm.get_results(min_max_scale=False)
    bmdf = bmdf.reset_index(drop=False,)
    bmdf.to_csv(output_path, index=False,)