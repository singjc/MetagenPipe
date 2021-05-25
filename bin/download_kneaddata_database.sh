#!/bin/bash
# Note: This is to be run in microbiome_ubunut20 Docker container
source /root/anaconda/etc/profile.d/conda.sh
conda init
output_dir='/project/data/raw/kneaddata_db'
mkdir $output_dir
cd $output_dir
conda activate microbiome
kneaddata_database --download human_genome bowtie2 $output_dir
conda deactivate
