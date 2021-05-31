#!/bin/bash
# Note: This is to be run in microbiome_ubunut20 Docker container
output_dir='/project/data/raw/kraken2_db'
mkdir $output_dir
kraken2-build --standard --threads 16 --db $output_dir
