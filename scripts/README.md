# Scripts

This folder contains standalone Python scripts that are used by the Snakemake workflow but can also be executed on their own.

- **`train_model.py`** – preprocesses an AnnData object, trains scVI and SCANVI models and writes the resulting embeddings, models and metrics to the output directory. The script expects three arguments: the input `.h5ad` file, the number of highly variable genes and the output directory.
- **`benchmark.py`** – benchmarks the latent representations of an integrated dataset using metrics from `scib-metrics`. It writes a CSV file with the results.
- **`extract_deg.py`** – loads a trained SCANVI model and performs differential expression testing, writing the results to CSV.

All scripts assume that the required Python packages listed in `environment.yml` are available.
