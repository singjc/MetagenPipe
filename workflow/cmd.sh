#!/bin/usr/sh
# set for failure on any error
set -e
# Activate Conda env
source /root/anaconda/etc/profile.d/conda.sh
conda activate microbiome

# SRA Download Workflow
#snakemake --snakefile Snakefile.sradownload_wf -j 2

#snakemake --snakefile Snakefile.sradownload_subs_wf -j 1

# Preprocessing Workflow

#snakemake --snakefile Snakefile.subsample_wf -j 6
# to unlock directory
snakemake --snakefile Snakefile.subsample_kraken2_wf --unlock True

# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 6  --config master_output_dir='kraken2_10M' reads_subsample=10000000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 6  --config master_output_dir='kraken2_5M' reads_subsample=5000000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 6  --config master_output_dir='kraken2_1M' reads_subsample=1000000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 6  --config master_output_dir='kraken2_500K' reads_subsample=500000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 6  --config master_output_dir='kraken2_100K' reads_subsample=100000
snakemake --snakefile Snakefile.subsample_kraken2_wf -j 6  --config master_output_dir='kraken2_50K' reads_subsample=50000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 6  --config master_output_dir='kraken2_25K' reads_subsample=25000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 6  --config master_output_dir='kraken2_10K' reads_subsample=10000

conda deactivate
