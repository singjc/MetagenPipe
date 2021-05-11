#!/bin/bash
set -e
# wget for downloading files
#apt-get install wget
# inetall libtbb2, bowtie2 depends on this
apt-get install libtbb2
# install java
dpkg -i /misc_files/jdk-15.0.2_linux-x64_bin.deb
# install sra toolkit
wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/2.10.9/sratoolkit.2.10.9-ubuntu64.tar.gz
tar -xvzf sratoolkit.2.10.9-ubuntu64.tar.gz
mv sratoolkit.2.10.9-ubuntu64 /root/
echo 'export PATH="$PATH:/root/sratoolkit.2.10.9-ubuntu64/bin"' >> /root/.bashrc
/root/sratoolkit.2.10.9-ubuntu64/bin/vdb-config --restore-defaults
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
# installs wrapper for fastq dumbps
conda install -y -c bioconda bioinfokit
# installs kneaddata for preprocessing
conda install -c bioconda kneaddata
# other useful packages
conda install numpy -y
conda install scipy -y
conda install pandas -y
conda install seaborn -y
conda install scikit-learn -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge scikit-plot -y
conda install -c conda-forge xgboost -y
conda install jupyter -y
conda install openpyxl -y
conda install pytorch -y
conda install -c anaconda beautifulsoup4 -y
conda install lxml -y
conda install -c bioconda seqtk -y
conda install -c conda-forge mamba -y
conda install -c conda-forge bioconda::snakemake -y
# metaphlan + kneaddata install databases
mkdir /databases
metaphlan --install --index mpa_v30_CHOCOPhlAn_201901 --bowtie2db /databases/metaphlan/
kneaddata_database --download human_genome bowtie2 /databases/kneaddata_human_bowtie2
conda deactivate

