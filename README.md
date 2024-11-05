# scVI and SCANVI Training Pipeline

This repository provides a Python script for training and evaluating scVI and SCANVI models on single-cell RNA sequencing data. The script includes functionality for data preprocessing, model training, and result analysis.

## Features

* **Data Preprocessing:** Filters cells and genes, structures annotation labels, and removes mitochondrial genes.
* **Model Training:** Trains scVI and SCANVI models with customizable parameters.
* **Result Analysis:**  Extracts training metrics, performs differential expression analysis, and generates latent representations.
* **Query Mapping:** Maps query datasets to the reference latent space using trained models.
* **Integration Benchmarking:**  Evaluates the performance of different integration methods (PCA, Scanorama, Harmony, scVI, SCANVI).

## Requirements

* Python 3.8+
* scVI
* scanpy
* anndata
* scib-metrics
* sklearn
* torch
