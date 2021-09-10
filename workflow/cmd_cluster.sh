#!/bin/usr/sh
# set for failure on any error
set -e

top_results_dir="results"
log_dir="logs"
archive_results=0
timestamp=$(date +"%Y_%b_%d_%H_%M_%S")

top_archive_results_dir="archive"

if [ ! -d $top_archive_results_dir ]; then
  mkdir $top_archive_results_dir
fi
if [ -d $top_results_dir ]
then
  if [ $archive_results -eq 1 ]
  then
    archive_results_dir="${top_archive_results_dir}/${top_results_dir}_${timestamp}"
    echo "moving ${top_results_dir} contents to ${archive_results_dir}"
    mv $top_results_dir $archive_results_dir
  fi
fi
if [ ! -d $top_results_dir ]
then
	echo "making top results directory"
	mkdir $top_results_dir
fi
if [ -d $log_dir ]
then
  if [ $archive_results -eq 1 ]
  then
    archive_log_dir="${top_archive_results_dir}/${log_dir}_${timestamp}"
    echo "moving ${log_dir} contents to ${archive_log_dir}"
    mv $log_dir $archive_log_dir
  fi
else
  mkdir $log_dir
fi

# Activate Conda env
# source /root/anaconda/etc/profile.d/conda.sh
# conda activate microbiome

#################################################
##            SRA Download Workflow            
#################################################
#snakemake --snakefile Snakefile.sradownload_wf -j 2

#################################################
##            Preprocessing Workflow
#################################################

## Cluster WF

## **NOTE**: When running on the cluster, src/ and data/ has to be moved to workflow/ because for some reason even if you use absolute paths, it fails to find the necessary files if it's not relative to current working directory (i.e. workflow)
## **NOTE**: You may also want to build the docker image locally using singularity, and then cp the image to the cluster. The cluster may not have internet, or depending on how large the image is, it might take a long time or cause pinging issues on the cluster
##
## Building local image using singularity
##
## 1. First ensure you have singulatiry installed.https://sylabs.io/guides/3.0/user-guide/quick_start.html
## 2. sudo singularity build florabioworks.simg docker://whitleyo/microbiome_ubuntu20:cluster
##      # note: use sudo to build the singularity to ensure it has the right permissions.
## 3. scp florabioworks.simg <hpc-cluster>:/cluster/home/microbiome_OJS/workflow/.snakemake/singularity/
##      # note: assuming there is a .snakemake folder, usually produced when running snakemake once

# activate python to run snakemake
source /project/6011811/bin/pyenv38/bin/activate

# Load singularity for docker containers
## You may need to load the singularity module on the cluster
# module load singularity/3.8
 
# Snakemake slurm config profile
snakemake_config=config/slurm_cluster

# Global configs
## Set raw data path
raw_data_dir="data/raw/SRA/"

## to run at various depths

#########################
##  10M
#########################
snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=32 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10M" reads_subsample=10000000 --use-singularity

snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=32 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10M" reads_subsample=10000000 --use-singularity --unlock -R

snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=32 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10M" reads_subsample=10000000 --use-singularity

snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=32 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10M" reads_subsample=10000000 --use-singularity --unlock -R

snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=32 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10M" reads_subsample=10000000 --use-singularity
# machine learning notebook workflow automated through papermill
snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_10M" master_output_dir="${top_results_dir}/kraken2_PE_10M" --use-singularity

##########################
###  5M
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=64 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_5M" reads_subsample=5000000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=64 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_5M" reads_subsample=5000000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=64 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_5M" reads_subsample=5000000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_5M" master_output_dir="${top_results_dir}/kraken2_PE_5M" --use-singularity
#
##########################
###  1M
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=47 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_1M" reads_subsample=1000000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=47 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_1M" reads_subsample=1000000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=47 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_1M" reads_subsample=1000000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_1M" master_output_dir="${top_results_dir}/kraken2_PE_1M" --use-singularity
#
##########################
###  500K
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=74 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_500K" reads_subsample=500000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=74 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_500K" reads_subsample=500000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=74 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_500K" reads_subsample=500000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_500K" master_output_dir="${top_results_dir}/kraken2_PE_500K" --use-singularity
#
##########################
###  100K
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=43 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_100K" reads_subsample=100000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=43 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_100K" reads_subsample=100000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=43 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_100K" reads_subsample=100000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_100K" master_output_dir="${top_results_dir}/kraken2_PE_100K" --use-singularity
#
##########################
###  50K
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=86 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_50K" reads_subsample=50000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=86 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_50K" reads_subsample=50000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=86 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_50K" reads_subsample=50000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_50K" master_output_dir="${top_results_dir}/kraken2_PE_50K" --use-singularity
#
##########################
###  25K
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=103 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_25K" reads_subsample=25000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=103 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_25K" reads_subsample=25000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=103 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_25K" reads_subsample=25000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_250K"master_output_dir="${top_results_dir}/kraken2_PE_25K" --use-singularity
#
##########################
###  10K
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=207 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=207 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=207 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K" reads_subsample=10000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_10K" master_output_dir="${top_results_dir}/kraken2_PE_10K" --use-singularity
#
##########################
###  5K
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=23 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_5K" reads_subsample=5000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=23 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_5K" reads_subsample=5000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=23 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_5K" reads_subsample=5000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_5K" master_output_dir="${top_results_dir}/kraken2_PE_5K" --use-singularity
#
##########################
###  1K
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=809 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_1K" reads_subsample=1000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=809 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_1K" reads_subsample=1000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=809 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_1K" reads_subsample=1000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_1K" master_output_dir="${top_results_dir}/kraken2_PE_1K" --use-singularity
#
##########################
###  500
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=42 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_500" reads_subsample=500 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=42 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_500" reads_subsample=500 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=42 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_500" reads_subsample=500 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_500" master_output_dir="${top_results_dir}/kraken2_PE_500" --use-singularity
#
#
### run at 10K with various subsample seeds to see if we get variability in number of features detected
##########################
###  seed42
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=42 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed42" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=42 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed42" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=42 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed42" reads_subsample=10000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_10K_seed42" master_output_dir="${top_results_dir}/kraken2_PE_10K_seed42" --use-singularity
#
##########################
###  seed47
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=47 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed47" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=47 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed47" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=47 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed47" reads_subsample=10000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_10K_seed47" master_output_dir="${top_results_dir}/kraken2_PE_10K_seed47" --use-singularity
#
##########################
###  seed29
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=29 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed29" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=29 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed29" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=29 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed29" reads_subsample=10000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_10K_seed29" master_output_dir="${top_results_dir}/kraken2_PE_10K_seed29" --use-singularity
#
##########################
###  seed93
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=93 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed93" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=93 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed93" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=93 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed93" reads_subsample=10000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_10K" master_output_dir="${top_results_dir}/kraken2_PE_10K_seed93" --use-singularity
#
##########################
###  seed87
##########################
#snakemake --profile $snakemake_config --snakefile Snakefile.subsample_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=87 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed87" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kneaddata_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=87 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed87" reads_subsample=10000 --use-singularity
#
#snakemake --profile $snakemake_config --snakefile Snakefile.kraken_PE_cluster -j 16 --config raw_data_dir=$raw_data_dir seqtk_seed=87 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE_10K_seed87" reads_subsample=10000 --use-singularity
## machine learning notebook workflow automated through papermill
#snakemake --snakefile Snakefile.papermill_compile_cluster -j 1 --config raw_data_dir=$raw_data_dir exp_dir="kraken2_PE_10K_seed87" master_output_dir="${top_results_dir}/kraken2_PE_10K_seed87" --use-singularity
#
