# scVI and SCANVI Training Pipeline

This repository contains a Snakemake workflow and helper scripts for training [scVI](https://scvi-tools.org/) and SCANVI models on single-cell RNA sequencing data.
The pipeline performs data preprocessing, model training, benchmarking and differential expression analyses.

## Features

* **Data preprocessing** – filters cells and genes, manages annotation labels and removes mitochondrial genes.
* **Model training** – trains scVI and SCANVI models with customizable parameters and checkpointing.
* **Query mapping** – maps query data sets into a trained reference latent space.
* **Benchmarking** – evaluates embeddings using PCA, Scanorama, Harmony, scVI and SCANVI.
* **Result extraction** – exports training metrics and differential expression results.

## Installation

Create the conda environment used by the workflow:

```bash
mamba env create -f environment.yml
```

Activate the environment before running the pipeline.

## Usage

Edit `config/config.yaml` to point to your input AnnData file and desired output directory.
The workflow can then be executed locally with

```bash
snakemake --use-conda --cores 4 -s Snakefile
```

For HPC execution a Slurm profile is provided under `config/gpu` (see the README in that directory).

Individual steps can also be run directly via the scripts in `scripts/`.

## Repository layout

* `scripts/` – Python scripts used in the workflow.
* `config/` – configuration files and cluster profiles.
* `notebooks/` – example notebooks exploring various parts of the pipeline.

