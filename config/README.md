# Configuration Files

This directory holds configuration files used by the Snakemake workflow.

- **`config.yaml`** – main configuration file specifying the `input_adata` path and the `output_path` directory where results will be written.
- **`gpu.json`** – Slurm resource settings used by the cluster profile in `config/gpu`.
- **`gpu/`** – Snakemake profile for running the workflow on an HPC cluster (see `config/gpu/README.md`).

Edit `config.yaml` before running the pipeline to match your local paths.
