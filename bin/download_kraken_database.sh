#!/bin/bash
# Note: This is to be run in microbiome_ubunut20 Docker container
output_dir='../data/raw/kraken2_db'
mkdir $output_dir
cd $output_dir
wget https://genome-idx.s3.amazonaws.com/kraken/k2_standard_20201202.tar.gz 
tar -xvzf k2_standard_20201202.tar.gz
rm k2_standard_20201202.tar.gz
