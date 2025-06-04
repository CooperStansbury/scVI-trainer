# GPU Snakemake Profile

The files in this directory define a Snakemake profile for running the workflow on a Slurm GPU cluster.

- **`config.yaml`** sets default Snakemake command line options such as the cluster submission command and number of cores.
- **`../gpu.json`** contains the resource specifications passed to `sbatch` (memory, wall time, number of GPUs, etc.).

To launch the workflow with this profile use:

```bash
snakemake --profile config/gpu --use-conda --cores 36 -s workflow.smk
```

The `gpu_launcher.sh` script provides an example batch submission wrapper that copies the current `Snakefile` to `workflow.smk` before execution.
