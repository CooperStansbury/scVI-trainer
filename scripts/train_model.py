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


def preprocess_adata(adata, min_genes=1000, min_cells=250):
    """
    Preprocesses an AnnData object by filtering cells and genes, 
    structuring annotation labels, and removing mitochondrial genes.

    Args:
        adata: AnnData object to be preprocessed.
        min_genes: Minimum number of genes expressed in a cell to pass filtering.
        min_cells: Minimum number of cells expressing a gene to pass filtering.

    Returns:
        AnnData: Preprocessed AnnData object.
    """

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Structure the annotation labels
    adata.obs['cell_label'] = adata.obs['standard_cell_type'].apply(lambda x: str(x).strip())
    adata = adata[adata.obs['cell_label'].notna(), :].copy()  
    categories = list(adata.obs["cell_label"].unique()) + ["Unknown"]
    cat_dtype = pd.CategoricalDtype(categories=categories)
    adata.obs["cell_label"] = adata.obs["cell_label"].astype(cat_dtype)

    # Filter out mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata = adata[:, ~adata.var['mt']].copy()

    return adata


def get_datasplit(adata, query_data=['iHSC']):
  """Splits AnnData by 'dataset'.

  Args:
    adata: AnnData object.
    query_data: Values to match in `adata.obs['dataset']`.

  Returns:
    tuple: (AnnData without matches, AnnData with matches)
  """
  rdata = adata[~adata.obs['dataset'].isin(query_data), :].copy()
  qdata = adata[adata.obs['dataset'].isin(query_data), :].copy()
  return rdata, qdata
    

def check_directory(dir_path):
    """
    Checks for the existence of specific subdirectories within a given directory 
    and creates them if they don't exist.

    Args:
      dir_path: The path to the main directory.
    """

    required_subdirs = [
        "models", 
        "training_metrics", 
        "benchmarks", 
        "results",
        "imputed_adata", 
        "checkpoints",
    ]

    for subdir in required_subdirs:
        subdir_path = os.path.join(dir_path, subdir)
        if not os.path.exists(subdir_path):
            print(f"Subdirectory '{subdir_path}' does not exist. Creating it...")
            os.makedirs(subdir_path)
            print(f"Subdirectory '{subdir_path}' successfully created.")


def get_loss_history(model):
  """Extracts and formats the loss history from a trained scVI model.

  Args:
    model: A trained scVI model.

  Returns:
    pd.DataFrame: A DataFrame containing the training and validation 
                  loss history, formatted for plotting.
  """

  metrics = pd.concat(model.history.values(), ignore_index=False, axis=1)
  metrics = metrics[metrics["validation_loss"].notna()]
  metrics = metrics.reset_index(drop=False, names="epoch")
  return metrics


def check_directory(dir_path):
    """
    Checks for the existence of specific subdirectories within a given directory 
    and creates them if they don't exist.

    Args:
      dir_path: The path to the main directory.
    """

    required_subdirs = [
        "models", 
        "training_metrics", 
        "benchmarks", 
        "results",
        "imputed_adata", 
        "model_checkpoints",
    ]

    for subdir in required_subdirs:
        subdir_path = os.path.join(dir_path, subdir)
        if not os.path.exists(subdir_path):
            print(f"Subdirectory '{subdir_path}' does not exist. Creating it...")
            os.makedirs(subdir_path)
            print(f"Subdirectory '{subdir_path}' successfully created.")


def get_loss_history(model):
  """Extracts and formats the loss history from a trained scVI model.

  Args:
    model: A trained scVI model.

  Returns:
    pd.DataFrame: A DataFrame containing the training and validation 
                  loss history, formatted for plotting.
  """

  metrics = pd.concat(model.history.values(), ignore_index=False, axis=1)
  metrics = metrics[metrics["validation_loss"].notna()]
  metrics = metrics.reset_index(drop=False, names="epoch")
  return metrics





def train_scvi_model(
    adata,
    batch_key="dataset",
    layer="counts",
    labels_key="cell_label",
    epochs=400,
    n_latent=40,
    n_hidden=256,
    dropout_rate=0.2,
    n_layers=3,
    batch_size=10000,
    plan_kwargs=None,  
    dirpath=None,
):
    """Trains an scVI model on the provided AnnData object.
    
    This function sets up and trains an scVI model with specified parameters, 
    including data setup, model architecture, training plan, and checkpointing.
    
    Args:
        adata (AnnData): 
            AnnData object containing the single-cell data.
        batch_key (str, optional): 
            Key in `adata.obs` for batch information. Defaults to "dataset".
        layer (str, optional): 
            Key in `adata.layers` for count data. Defaults to "counts".
        labels_key (str, optional): 
            Key in `adata.obs` for cell labels. Defaults to "cell_label".
        epochs (int, optional): 
            Maximum number of training epochs. Defaults to 400.
        n_latent (int, optional): 
            Dimensionality of the latent space. Defaults to 40.
        n_hidden (int, optional): 
            Number of nodes per hidden layer. Defaults to 256.
        dropout_rate (float, optional): 
            Dropout rate for the model. Defaults to 0.2.
        n_layers (int, optional): 
            Number of layers in the encoder and decoder. Defaults to 3.
        batch_size (int, optional): 
            Batch size for training. Defaults to 10000.
        plan_kwargs (dict, optional): 
            Dictionary of keyword arguments for the training plan. 
            Defaults to None, in which case a default plan is used.
        dirpath (str, optional): 
            Directory path for saving model checkpoints. Defaults to None.
    
    Returns:
        scvi.model.SCVI: The trained scVI model.
    """
    scvi.model.SCVI.setup_anndata(
      adata, 
      batch_key=batch_key, 
      layer=layer, 
      labels_key=labels_key
    )
    
    torch.cuda.empty_cache()
    
    model = scvi.model.SCVI(
      adata,
      use_layer_norm="both",
      use_batch_norm="none",
      n_latent=n_latent,
      n_hidden=n_hidden,
      encode_covariates=True,
      dropout_rate=dropout_rate,
      n_layers=n_layers,
    )
    
    start_time = time.time()
    
    checkpointer = scvi.train.SaveCheckpoint(
        dirpath=dirpath,
        filename="scvi_{epoch}-{step}-{monitor}",
        monitor='validation_loss',
        load_best_on_end=True,
    )
    
    model.train(
        max_epochs=epochs,
        accelerator="gpu",
        devices="auto",
        enable_model_summary=True,
        batch_size=batch_size,
        load_sparse_tensor=True,
        plan_kwargs=plan_kwargs,
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_monitor='elbo_validation',
        enable_checkpointing=True,
        callbacks=[checkpointer],
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Training completed in {total_time:.2f} seconds")
    
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"Training time: {minutes} minutes {seconds} seconds")
    
    return model


def train_scanvi_model(
    scvi_model,
    unlabeled_category="Unknown",
    epochs=400,
    batch_size=10000,
    plan_kwargs=None,
    dirpath=None,
):
    """Trains a SCANVI model using a pre-trained scVI model.
    
    This function takes a trained scVI model and uses it to initialize and train
    a SCANVI model for semi-supervised learning on single-cell data with 
    labeled and unlabeled cell types.
    
    Args:
        scvi_model (scvi.model.SCVI): 
            A trained scVI model to initialize SCANVI.
        unlabeled_category (str, optional): 
            Category name for unlabeled cells in `adata.obs`. 
            Defaults to "Unknown".
        epochs (int, optional): 
            Maximum number of training epochs. Defaults to 400.
        batch_size (int, optional): 
            Batch size for training. Defaults to 10000.
        plan_kwargs (dict, optional): 
            Dictionary of keyword arguments for the training plan. 
            Defaults to None, in which case a default plan is used.
        dirpath (str, optional): 
            Directory path for saving model checkpoints. Defaults to None.
    
    Returns:
        scvi.model.SCANVI: The trained SCANVI model.
    """
    torch.cuda.empty_cache()
    
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model, 
        unlabeled_category=unlabeled_category,
    )
    
    start_time = time.time()

    checkpointer = scvi.train.SaveCheckpoint(
        dirpath=dirpath,
        filename="scanvi_{epoch}-{step}-{monitor}",
        monitor='validation_loss',
        load_best_on_end=True,
    )
    
    scanvi_model.train(
        max_epochs=epochs,
        accelerator="gpu",
        devices="auto",
        enable_model_summary=True,
        early_stopping=True,
        batch_size=batch_size,
        plan_kwargs=plan_kwargs,
        early_stopping_patience=5,
        callbacks=[checkpointer],
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Training completed in {total_time:.2f} seconds")
    
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"Training time: {minutes} minutes {seconds} seconds")
    
    return scanvi_model


        
if __name__ == "__main__":
    torch.cuda.empty_cache()
    adata_path = sys.argv[1]
    n_hvg = sys.argv[2]
    output_dir = sys.argv[3]

    EPOCHS = 500

    print(f"\n--- Running train_model.py ---")  # Added print statement
    print(f"adata_path: {adata_path}")  # Added print statement
    print(f"n_hvg: {n_hvg}")  # Added print statement
    print(f"output_dir: {output_dir}")  # Added print statement

    # handle output path
    check_directory(output_dir)

    # name the model
    model_name = f"hvg{n_hvg}"

    # load input data
    adata = sc.read_h5ad(adata_path)
    adata.X = adata.layers['counts'].copy() # TODO: don't hardcode thie
    print(f"\n------------ raw data ------------")
    print(adata)
    
    sc.logging.print_memory_usage()

    """
    Filtering:
        - low quality cells/genes
        - missing annotations
        - mitochondrial genes
    """

    print(f"--- FILTERING DATA ---")
    MIN_GENES = 1000
    MIN_CELLS = 500
    
    adata = preprocess_adata(
        adata, 
        min_genes=MIN_GENES,
        min_cells=MIN_CELLS,
    )
    
    # split data into query and reference sets
    rdata, qdata = get_datasplit(adata)

    # make annotations
    qdata.obs["cell_label"] = 'Unknown'
    print(qdata.obs["cell_label"].value_counts())
    print()

    print(f"--- SELECTING FEATURES ---")
    sc.pp.highly_variable_genes(
        rdata, 
        n_top_genes=int(n_hvg),
        flavor="seurat_v3",
        subset=True,
        batch_key='dataset',
    )

    """
    scVI MODEL
    """
    print(f"--- TRAINNING scVI MODEL ---")
    plan_kwargs = {
          "lr": 0.001,
          "n_epochs_kl_warmup": 10,
          "reduce_lr_on_plateau": True,
          "lr_patience": 8,
          "lr_factor": 0.1,
      }

    scvi_model = train_scvi_model(
        rdata,
        plan_kwargs=plan_kwargs,
        epochs=EPOCHS,
        dirpath=f"{output_dir}checkpoints/",
    )

    # store the model
    outpath = f"{output_dir}models/"
    scvi_model.save(
        outpath, 
        overwrite=True, 
        save_anndata=True,
        prefix=f"scvi_{model_name}_"
    ) 

    # store the loss metrics
    outpath = f"{output_dir}training_metrics/{model_name}_scvi.csv"
    scvi_metrics = get_loss_history(scvi_model)
    scvi_metrics.to_csv(outpath, index=False)

    """
    scANVI MODEL
    """
    print(f"--- TRAINNING scANVI MODEL ---")
    scanvi_model = train_scanvi_model(
        scvi_model,
        plan_kwargs=plan_kwargs,
        epochs=EPOCHS,
        dirpath=f"{output_dir}checkpoints/",
    )

    # store the model
    outpath = f"{output_dir}models/"
    scanvi_model.save(
        outpath, 
        overwrite=True, 
        save_anndata=True,
        prefix=f"scanvi_{model_name}_"
    ) 

    # store the loss metrics
    outpath = f"{output_dir}training_metrics/{model_name}_scanvi.csv"
    scanvi_metrics = get_loss_history(scanvi_model)
    scanvi_metrics.to_csv(outpath, index=False)

    """
    Differential Expression
    """
    print(f"--- DEG TESTING ---")
    torch.cuda.empty_cache()
    deg = scanvi_model.differential_expression(
        rdata,
        groupby='cell_label',
        batch_correction=True,
        filter_outlier_cells=True,
    )
    outpath = f"{output_dir}results/{model_name}_scanvi_DEG.csv"
    deg = deg.reset_index()
    deg.to_csv(outpath, index=False,)

    """
    Latent Representations and Counts
    """
    print(f"--- COMPUTING EMBEDDINGS AND COUNTS ---")
    rdata.obsm['X_scvi'] = scvi_model.get_latent_representation()
    rdata.obsm['X_scanvi'] = scanvi_model.get_latent_representation()

    # get batch-corrected counts
    rdata.layers['scvi_counts'] = scvi_model.get_normalized_expression(return_mean=False)
    rdata.layers['scanvi_counts'] = scanvi_model.get_normalized_expression(return_mean=False)

    """
    scVI Query Mapping
    """
    print(f"--- QUERY MAPPING (scVI) ---")
    scvi.model.SCVI.prepare_query_anndata(
        qdata, 
        scvi_model,
    )
    
    scvi_query = scvi.model.SCVI.load_query_data(
        qdata, 
        scvi_model,
    )
    
    scvi_query.train(
        max_epochs=EPOCHS, 
        plan_kwargs=plan_kwargs,
        early_stopping=True,
        early_stopping_monitor='elbo_validation',
        early_stopping_patience=5,
    )
    
    qdata.obsm['X_scvi'] = scvi_query.get_latent_representation()
    qdata.layers['scvi_counts'] = scvi_query.get_normalized_expression(return_mean=False)

    """
    scANVI Query Mapping
    """
    print(f"--- QUERY MAPPING (scANVI) ---")
    scvi.model.SCANVI.prepare_query_anndata(
        qdata, 
        scanvi_model,
    )

    scanvi_query = scvi.model.SCANVI.load_query_data(
        qdata, 
        scanvi_model,
    )

    scanvi_query.train(
        max_epochs=EPOCHS, 
        plan_kwargs=plan_kwargs,
        early_stopping=True,
        early_stopping_monitor='elbo_validation',
        early_stopping_patience=5,
    )

    qdata.obsm['X_scanvi'] = scanvi_query.get_latent_representation()
    qdata.layers['scanvi_counts'] = scanvi_query.get_normalized_expression(return_mean=False)

    # store predictions
    outpath = f"{output_dir}results/{model_name}_predictions.csv"
    pred_proba = scanvi_query.predict(soft=True)
    pred_proba['prediction'] = pred_proba.idxmax(axis=1)
    pred_proba = pred_proba.reset_index(drop=False, names='cell_id')
    pred_proba.to_csv(outpath, index=False,)

    """
    Integrate and Store
    """
    idata = an.concat([rdata, qdata], label="batch")

    # sc.pp.subsample(idata, n_obs=10000)
    # sorted_indices = idata.obs['dataset'].sort_values().index 
    # idata = idata[sorted_indices, :].copy()

   # build some convenience attributes
    start_time = time.time()
    sc.pp.pca(
        idata,
        layer='counts',
        n_comps=25,
    )
    end_time = time.time()
    print(f"PCA completed in {end_time - start_time:.2f} seconds")
    
    print(f"--- HARMONIZATION AND OUTPUT ---")
    start_time = time.time()
    sce.pp.scanorama_integrate(
        idata, 
        key='dataset', 
        adjusted_basis='X_scanorama',
        verbose=1,
    )
    end_time = time.time()
    print(f"Scanorama integration completed in {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    sce.pp.harmony_integrate(
        idata, 
        key='dataset', 
        adjusted_basis='X_harmony',
    )
    end_time = time.time()
    print(f"Harmony integration completed in {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    sc.pp.neighbors(
        idata, 
        use_rep='X_scanvi',
    )
    end_time = time.time()
    print(f"Neighbor computation completed in {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    sc.tl.umap(
        idata, 
        min_dist=0.25, 
        method='rapids',
    )
    end_time = time.time()
    print(f"UMAP embedding completed in {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    outpath = f"{output_dir}imputed_adata/{model_name}.h5ad"
    idata.write(outpath)
    end_time = time.time()
    print(f"Saving AnnData completed in {end_time - start_time:.2f} seconds")




    






    



    


    
    
    
    