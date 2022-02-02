#!/bin/usr/sh
# set for failure on any error
set -e
# Activate Conda env
# source /src/anaconda/etc/profile.d/conda.sh
# conda activate microbiome



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

# SRA Download Workflow
# snakemake --snakefile Snakefile.sradownload_wf -j 2
#snakemake --snakefile Snakefile.sradownload_subs_wf -j 2


# Preprocessing Workflow

# snakemake --snakefile Snakefile.subsample_kneaddata_PE_wf -j 2  --config seqtk_seed=32 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kneaddata_PE" reads_subsample=100000
#snakemake --snakefile Snakefile.subsample_kneaddata_PE_wf -j 2  --config seqtk_seed=32 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kneaddata_PE_microbiome_ubuntu20" reads_subsample=100000 
# snakemake --snakefile Snakefile.subsample_kraken2_PE_wf -j 2  --config seqtk_seed=32 nthreads=2 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_PE" reads_subsample=50000
#snakemake --snakefile Snakefile.subsample_wf -j 6
# to unlock directory
# snakemake --snakefile Snakefile.subsample_kraken2_wf --unlock True
# metaphlan parser
# snakemake --snakefile Snakefile.metaphlan_parse_wf -j 1

# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=32 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_10M" reads_subsample=10000000

# humann3 workflow
# snakemake --snakefile Snakefile.subsample_humann3_PE_wf -j 2
# transform humann3 outputs
snakemake --snakefile Snakefile.humann3_matrix_transform_wf -j 1

# to run kraken2 at various depths

# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=32 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_10M" reads_subsample=10000000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=64 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_5M" reads_subsample=5000000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=47 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_1M" reads_subsample=1000000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=74 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_500K" reads_subsample=500000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=43 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_100K" reads_subsample=100000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=86 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_50K" reads_subsample=50000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=103 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_25K" reads_subsample=25000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=207 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_10K" reads_subsample=10000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=23 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_5K" reads_subsample=5000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=809 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_1K" reads_subsample=1000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=42 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_500" reads_subsample=500
# # run at 10K with various subsample seeds to see if we get variability in number of features detected
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=42 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_10K_seed42" reads_subsample=10000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=47 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_10K_seed47" reads_subsample=10000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=29 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_10K_seed29" reads_subsample=10000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=93 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_10K_seed93" reads_subsample=10000
# snakemake --snakefile Snakefile.subsample_kraken2_wf -j 8  --config seqtk_seed=87 log_dir=$log_dir master_output_dir="${top_results_dir}/kraken2_10K_seed87" reads_subsample=10000



# machine learning notebook workflow automated through papermill
# snakemake --snakefile Snakefile.papermill_compile_wf -j 1
# papermill workflow with data preparation + analyses separated
# snakemake --snakefile Snakefile.papermill_compile_analysis_suite_wf -j 1
# snakemake --snakefile Snakefile.papermill_compile_analysis_suite_wf -j 1 --config inp_mat_file='/project/data/raw/jie_fulldata/matrix_kraken/kraken_freq_mat_fulldata.csv' master_output_dir='/project/workflow/results/papermill_jie_kraken_full'
# snakemake --snakefile Snakefile.papermill_compile_analysis_suite_wf -j 1 --config inp_mat_file='/project/data/preprocessed/jie_full_data_metaphlan/freq_mat.csv' master_output_dir='/project/workflow/results/papermill_jie_metaphlan_full'

# conda deactivate

