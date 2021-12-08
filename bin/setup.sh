#!/bin/bash
set -e
# wget for downloading files
#apt-get install wget
# inetall libtbb2, bowtie2 depends on this
apt-get install libtbb2
# moved sra toolkit installation to dockerfile
# # install sra toolkit
# wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/2.10.9/sratoolkit.2.10.9-ubuntu64.tar.gz
# tar -xvzf sratoolkit.2.10.9-ubuntu64.tar.gz
# mv sratoolkit.2.10.9-ubuntu64 /src/
# echo 'export PATH="$PATH:/src/sratoolkit.2.10.9-ubuntu64/bin"' >> /root/.bashrc
# /src/sratoolkit.2.10.9-ubuntu64/bin/vdb-config --restore-defaults
# install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p /src/anaconda
eval "$(/src/anaconda/bin/conda shell.bash hook)"
conda init
# create microbiome environment
conda create -n microbiome python=3.7 -y
conda activate microbiome
# installs metaphlan for getting bacterial species frequencies 
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda install -c conda-forge mamba -y
mamba install  -y -c bioconda python=3.7 metaphlan
# installs wrapper for fastq dumbps
mamba install -y -c bioconda bioinfokit
# installs kneaddata for preprocessing
mamba install -c bioconda kneaddata
# >>>>>>> 4714571382e3946538c526bb6d4e7819b9bf1cbe
# installs humann2 for pan-genome profiling
mamba install humann -c biobakery -y

## installs kraken2
#git clone https://github.com/DerrickWood/kraken2
#mv kraken2 /src/kraken2
#bash /src/kraken2/install_kraken2.sh /src/kraken2
#cp /src/kraken2/kraken2{,-build,-inspect} $HOME/bin
# other useful packages
mamba install numpy -y
mamba install scipy -y
mamba install pandas -y
mamba install seaborn -y
mamba install scikit-learn -y
mamba install -c conda-forge matplotlib -y
mamba install -c conda-forge scikit-plot -y
mamba install -c conda-forge xgboost -y
mamba install jupyter -y
mamba install openpyxl -y
mamba install pytorch -y
mamba install -c anaconda beautifulsoup4 -y
mamba install lxml -y
mamba install -c bioconda seqtk -y
mamba install -c conda-forge bioconda::snakemake -y
mamba install -c conda-forge papermill -y

# metaphlan, humann2, + kneaddata install databases
# mkdir /databases
# metaphlan --install --index mpa_v30_CHOCOPhlAn_201901 --bowtie2db /databases/metaphlan/
# humann_databases --download chocophlan full /databases/humann --update-config yes
# humann_databases --download uniref uniref90_diamond /databases/humann --update-config yes
# humann_databases --download utility_mapping full /databases/humann --update-config yes
# kneaddata_database --download human_genome bowtie2 /databases/kneaddata_human_bowtie2

# force diamond version to that necessary for humann3
mamba install -c bioconda -y diamond=0.9.36
conda deactivate
# kraken2 database

