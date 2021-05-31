#!/bin/bash
# Note: This is to be run in microbiome_ubunut20 Docker container
source /root/anaconda/etc/profile.d/conda.sh
conda init
output_dir='../data/raw/humann_db'
conda activate microbiome
humann_databases --download chocophlan full $output_dir --update-config no
humann_databases --download uniref uniref90_diamond $output_dir --update-config no
humann_databases --download utility_mapping full $output_dir --update-config no
conda deactivate
