#!/bin/bash
# Note: This is to be run in microbiome_ubunut20 Docker container
source /root/anaconda/etc/profile.d/conda.sh                                                                            conda init
output_dir='/project/data/raw/metaphlan2_db'
mkdir $output_dir
cd $output_dir
conda activate microbiome
metaphlan --install --bowtie2db $output_dir
conda deactivate
