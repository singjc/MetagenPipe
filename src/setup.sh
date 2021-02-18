#!/bin/bash
set -e
# wget for downloading files
#apt-get install wget
# install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -y
# create microbiome environment
conda create -n microbiome python=3.7 -y
conda activate microbiome
# installs metaphlan for getting bacterial species frequencies 
conda install  -y -c bioconda python=3.7 metaphlan
# other useful packages
conda install numpy -y
conda install scipy -y
conda install pandas -y
conda install seaborn -y
conda install scikit-learn -y
conda install jupyter -y
conda deactivate
