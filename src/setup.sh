#!/bin/bash
set -e
# wget for downloading files
#apt-get install wget
# install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p /root/anaconda
eval "$(/root/anaconda/bin/conda shell.bash hook)"
conda init
# create microbiome environment
conda create -n microbiome python=3.7 -y
conda activate microbiome
# installs metaphlan for getting bacterial species frequencies 
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda install  -y -c bioconda python=3.7 metaphlan
# other useful packages
conda install numpy -y
conda install scipy -y
conda install pandas -y
conda install seaborn -y
conda install scikit-learn -y
conda install jupyter -y
conda deactivate

wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/2.10.9/sratoolkit.2.10.9-ubuntu64.tar.gz
tar -xvzf sratoolkit.2.10.9-ubuntu64.tar.gz
mv sratoolkit.2.10.9-ubuntu64 /root/
