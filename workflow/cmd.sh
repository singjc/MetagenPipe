#!/bin/usr/sh

# Activate Conda env
source /root/anaconda/etc/profile.d/conda.sh
conda activate microbiome

# SRA Download Workflow
#snakemake --snakefile Snakefile.sradownload_wf -j 2

# Preprocessing Workflow
snakemake --snakefile Snakefile.subsample_wf -j 6
