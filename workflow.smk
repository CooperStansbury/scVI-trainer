from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import re
import os
import sys
from tabulate import tabulate

BASE_DIR = Path(workflow.basedir)
configfile: str(BASE_DIR) + "/config/config.yaml"

# big picture variables
OUTPUT = config['output_path']

input_adata = config['input_adata']
output_adata = OUTPUT + "raw_anndata/adata.h5ad"
# n_hvg = np.linspace(1000, 15000, 29).astype(int)
n_hvg = [3000, 6000, 9000, 12000, 15000]

print("\n----- HVG VALUES -----")
for v in n_hvg:
    print('-', v, ' genes')

# print statements
print("\n----- CONFIG VALUES -----")
for key, value in config.items():
    print(f"{key}: {value}")
    

rule all:
    input:
        OUTPUT + "raw_anndata/adata.h5ad",
        expand(OUTPUT + "flags/{hvg}.done", hvg=n_hvg),
        expand(OUTPUT + "benchmarks/{hvg}_benchmark.csv", hvg=n_hvg),
      
        
rule gather:
    input:
        input_adata
    output:
        output_adata
    shell:
        """
        cp {input} {output}
        """


rule train_model:
    input:
        adata=OUTPUT + "raw_anndata/adata.h5ad",
    output:
        flag=touch(OUTPUT + "flags/{hvg}.done"),
    params:
        output_dir=OUTPUT
    conda:
        "scanpy"
    log:
        OUTPUT + "logs/hvg{hvg}.txt",
    shell:
        """
        python scripts/train_model.py {input.adata} {wildcards.hvg} {params.output_dir} > {log}
        """


rule benchmark_model:
    input:
        flag=OUTPUT + "flags/{hvg}.done"
    output:
        flag=OUTPUT + "benchmarks/{hvg}_benchmark.csv",
    conda:
        "scanpy"
    params:
        adata=OUTPUT + "imputed_adata/hvg{hvg}.h5ad",
    shell:
        """
        python scripts/benchmark.py {params.adata} {output}
        """













    