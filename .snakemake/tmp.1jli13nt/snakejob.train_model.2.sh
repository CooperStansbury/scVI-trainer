#!/bin/sh
# properties = {"type": "single", "rule": "train_model", "local": false, "input": ["/scratch/indikar_root/indikar1/shared_data/scanvi_models/raw_anndata/adata.h5ad"], "output": ["/scratch/indikar_root/indikar1/shared_data/scanvi_models/flags/1000.done"], "wildcards": {"hvg": "1000"}, "params": {"output_dir": "/scratch/indikar_root/indikar1/shared_data/scanvi_models/", "hvg": "1000"}, "log": [], "threads": 1, "resources": {"mem_mb": 16146, "mem_mib": 15399, "disk_mb": 16146, "disk_mib": 15399, "tmpdir": "<TBD>"}, "jobid": 2, "cluster": {"account_slurm": "indikar99", "partition": "gpu_mig40,gpu,spgpu", "jobname": "train_model.hvg=1000", "nodes": 1, "ntasks": 1, "gpus": 1, "cpus": 32, "mem": "100G", "walltime": "36:00:00", "email": "cstansbu@umich.edu", "mailon": "a", "mail_type": "FAIL", "jobout": "oe", "outfile": "/scratch/indikar_root/indikar1/cstansbu/run_logs/train_model.hvg=1000.out"}}
cd /home/cstansbu/git_repositories/scVI-trainer && /home/cstansbu/miniconda3/envs/snakemake/bin/python3.12 -m snakemake --snakefile '/home/cstansbu/git_repositories/scVI-trainer/workflow.smk' --target-jobs 'train_model:hvg=1000' --allowed-rules 'train_model' --cores 'all' --attempt 1 --force-use-threads  --resources 'mem_mb=16146' 'mem_mib=15399' 'disk_mb=16146' 'disk_mib=15399' --wait-for-files '/home/cstansbu/git_repositories/scVI-trainer/.snakemake/tmp.1jli13nt' '/scratch/indikar_root/indikar1/shared_data/scanvi_models/raw_anndata/adata.h5ad' --force --keep-target-files --keep-remote --max-inventory-time 0 --nocolor --notemp --no-hooks --nolock --ignore-incomplete --rerun-triggers 'mtime' 'input' 'params' 'code' 'software-env' --skip-script-cleanup  --use-conda  --conda-frontend 'mamba' --conda-base-path '/home/cstansbu/miniconda3' --wrapper-prefix 'https://github.com/snakemake/snakemake-wrappers/raw/' --latency-wait 90 --scheduler 'ilp' --scheduler-solver-path '/home/cstansbu/miniconda3/envs/snakemake/bin' --default-resources 'mem_mb=max(2*input.size_mb, 1000)' 'disk_mb=max(2*input.size_mb, 1000)' 'tmpdir=system_tmpdir' --mode 2 && touch '/home/cstansbu/git_repositories/scVI-trainer/.snakemake/tmp.1jli13nt/2.jobfinished' || (touch '/home/cstansbu/git_repositories/scVI-trainer/.snakemake/tmp.1jli13nt/2.jobfailed'; exit 1)

