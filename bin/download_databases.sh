#!/bin/bash
set -e
source /root/anaconda/etc/profile.d/conda.sh 
conda init
bash download_metaphlan_database.sh 
bash download_humann_database.sh 
bash download_kneaddata_database.sh 
#bash download_kraken_database.sh 
