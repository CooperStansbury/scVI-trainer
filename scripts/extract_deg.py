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
    model_dir = sys.argv[2]
    output_path = sys.argv[3]

    # load input data
    adata = sc.read_h5ad(adata_path)
    adata.X = adata.layers['counts'].copy() # TODO: don't hardcode thie
    print(f"\n------------ raw data ------------")
    sc.logging.print_memory_usage()
    print(adata)

    # filter out the query set
    adata = adata[adata.obs['cell_label'] != 'Unknown', :].copy()

    basename = os.path.basename(adata_path).replace(".h5ad", "")
    prefix = f"scanvi_{basename}_"

    # load the model
    model = scvi.model.SCANVI.load(
        model_dir, 
        adata=adata,
        prefix=prefix,
    )

    print(model)

    print(f"--- DEG TESTING ---")
    torch.cuda.empty_cache()
    deg = model.differential_expression(
        adata,
        groupby='cell_label',
        batch_correction=True,
        filter_outlier_cells=True,
    )
    deg = deg.reset_index()
    deg.to_csv(output_path, index=False,)