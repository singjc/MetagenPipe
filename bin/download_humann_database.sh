#!/bin/bash
# Note: This is to be run in microbiome_ubunut20 Docker container
output_dir='../data/raw/humann_db'
mkdir $output_dir
cd $output_dir
conda activate microbiome
humann_databases --download chocophlan full $output_dir --update-config no
humann_databases --download uniref uniref90_diamond $output_dir --update-config no
humann_databases --download utility_mapping full $output_dir --update-config no
conda deactivate
