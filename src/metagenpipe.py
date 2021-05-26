import pandas as pd
import numpy as np
import click
import sys
import os
import glob
import multiprocessing

from preprocessing.data_processing import seqtk_call, kneaddata_call, metaphlan_call, parse_metaphlan_file
from preprocessing.report import save_report
from util.SRADownLoad import RunAll


# Main Command Line Interface
@click.group(chain=True)
@click.version_option()
@click.pass_context
def cli( ctx ):
    '''
    Raw Data Processing for Microbiome Metagenomics Data
    '''
    ctx.ensure_object(dict)

# Main SRA Downloader
@cli.command()
@click.argument('all_expt_accs', nargs=-1, type=click.STRING)
@click.option('--download_dir', default=(os.getcwd()+"/SRA/"), show_default=True, type=str, help='Directory to store data.')
@click.option('--extra_args', type=click.STRING, help='Extra arguments to pass to fastq_dump, encapsulated as a string. i.e. \"-I --gzip --split-files\"')
def sra_downloader( all_expt_accs, download_dir, extra_args ):
    '''
    Main function call to SRA Downloader
    '''
    RunAll(all_expt_accs, download_dir, extra_args)

# Main Subsampling with seqtk
@cli.command()
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--subsample_fraction', default=50000, show_default=True, type=float, help='Data fraction used for subsampling.')
@click.option('--output_dir', default=(os.getcwd()+"/raw_subsampled/"), show_default=True, type=str, help='Directory to store results.')
@click.option('--two_pass_mode/--no-two_pass_mode', default=False, show_default=True, help='Enable 2 pass mode')
@click.option('--rng_seed', default=100, show_default=True, type=float, help='Random seed, remember to use the same random seed to keep pairing.')
@click.option('--add_file_tag/--no-add_file_tag', default=False, show_default=True, help='Add a tag to subsampled files, denoting subsampled file using x seed at n fraction.')
@click.option('--remove_untarred_fastq/--no-remove_untarred_fastq', default=True, show_default=True, help='Remove untarred fastq file, if a tarred fastq file was used.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def run_seqtk( fastq_files, subsample_fraction, output_dir, two_pass_mode, rng_seed, add_file_tag, remove_untarred_fastq, nthreads ):
    '''
    Main function call to subsample fastq files using seqtk
    '''
    
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")
    # Prepare args for parallel processing
    fastq_files = list(fastq_files)
    subsample_fraction_pool = [subsample_fraction] * len(fastq_files)
    output_dir_pool = [output_dir] * len(fastq_files)
    two_pass_mode_pool = [two_pass_mode] * len(fastq_files)
    rng_seed_pool = [rng_seed] * len(fastq_files)
    add_file_tag_pool = [add_file_tag] * len(fastq_files)
    remove_untarred_fastq_pool = [remove_untarred_fastq] * len(fastq_files)
    # Initiate a pool with nthreads for parallel processing
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( seqtk_call, zip(fastq_files, subsample_fraction_pool, output_dir_pool, two_pass_mode_pool, rng_seed_pool, add_file_tag_pool, remove_untarred_fastq_pool) )
    pool.close()
    pool.join()

# Main Data Kneading with kneaddata
@cli.command(name='run-kneaddata', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True
))
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--reference_db', type=click.Path(exists=True), help='Reference Database file.')
@click.option('--output_dir', default=(os.getcwd()+"/kneadeddata/"), show_default=True, type=str, help='Directory to store results.')
@click.option('--trimmomatic', default=(glob.glob("/root/anaconda/envs/**/trimmomatic-*/", recursive=True)[0]), show_default=True, type=str, help='Directory to store results.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
@click.option('--remove_untarred_fastq/--no-remove_untarred_fastq', default=True, show_default=True, help='Remove untarred fastq file, if a tarred fastq file was used.')
@click.option('--extra_args', type=click.STRING, help='Extra arguments to pass to kneaddata, encapsulated as a string. i.e. \"-q=phred64 --trimmomatic-options=SLIDINGWINDOW:4:20 --trimmomatic-options=MINLEN:50\"')
@click.pass_context
def run_kneaddata( ctx, fastq_files, reference_db, output_dir, trimmomatic, nthreads, remove_untarred_fastq, extra_args ):
    '''
    Main function call to process the data with kneaddata
    '''
    if extra_args is not None:
        extra_args = extra_args.split(" ")
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    # Prepare args for parallel processing
    fastq_files = list(fastq_files)
    reference_db_pool = [reference_db] * len(fastq_files)
    output_dir_pool = [output_dir] * len(fastq_files)
    trimmomatic_pool = [trimmomatic] * len(fastq_files)
    remove_untarred_fastq_pool = [remove_untarred_fastq] * len(fastq_files)
    extra_args_pool = [extra_args] * len(fastq_files)
    # Initiate a pool with nthreads for parallel processing
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( kneaddata_call, zip(fastq_files, reference_db_pool, output_dir_pool, trimmomatic_pool, remove_untarred_fastq_pool, extra_args_pool) )
    pool.close()
    pool.join()


# Main MetaPhlan

@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--input_type', default=1, show_default=True, type=str, help="one of 'fastq', 'bowtie2out'. if bowtie2 out, use input files must be alignemnts from bowtie2 see metaphlan3 documentation for more details.")
@click.option('--output_dir_bowtie', default=(os.getcwd()+"/metaphlan_bowtie2out/"), show_default=True, type=str, help='Directory to store bowtie2 results.')
@click.option('--output_dir_profile', default=(os.getcwd()+"/metaphlan_profiles/"), show_default=True, type=str, help='Directory to store species profile results.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def run_metaphlan( inp_files, input_type, output_dir_bowtie, output_dir_profile, nthreads ):
    '''
    Main function call to process the data with metaphlan
    '''
    if len(inp_files) < 1:
        click.ClickException("At least one input file needs to be provided.")

    for f in inp_files:
        metaphlan_call(f, input_type, output_dir_bowtie, output_dir_profile, nthreads)


@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output_dir', default=(os.getcwd()), show_default=True, type=str, help='Directory to store matrix output')
@click.option('--outfile', default=('relative_abundances.csv'), show_default=True, type=str, help='filename for matrix output')
def parse_metaphlan_multi( inp_files, output_dir='./', outfile='relative_abundances.csv' ):
    """
    Parse a list or iterable of metaphlan output files. Note that taxa may vary from file to file
    :param inp_files: input files. list or iterable of strings
    :param output_dir: output directory to store matrix
    :return: writes matrix to file
    """
    for i in range(len(inp_files)):
        input_file_i = inp_files[i]
        df_i = parse_metaphlan_file(input_file_i)
        if i == 0:
            # df_c = dataframe, combined
            df_c = df_i
        else:
            cols_c = df_c.columns.to_numpy().astype('str')
            cols_i = df_i.columns.to_numpy().astype('str')
            diff_c_i = np.setdiff1d(cols_c, cols_i)
            diff_i_c = np.setdiff1d(cols_i, cols_c)
            if len(diff_c_i) > 0:
                for j in len(diff_c_i):
                    diff_c_i_j = diff_c_i[j]
                    df_i[diff_c_i_j] = np.array([0.0], dtype='float32')

            if len(diff_i_c) > 0:
                for j in len(diff_i_c):
                    diff_i_c_j = diff_i_c[j]
                    df_c[diff_i_c_j] = np.array([0.0], dtype='float32')
            # put columns in correct order, then append.
            # by end of 2 if statements, both df_c and df_i should have
            # same columns
            sorted_cols = df_c.columns.to_numpy().astype('str').sort()
            df_c = df_c.loc[:, sorted_cols]
            df_i = df_i.loc[:, sorted_cols]
            df_c.append(df_i)

    df_c.to_csv(os.path.join(output_dir, outfile))

@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output_dir', default=(os.getcwd()), show_default=True, type=str, help='Directory to store report')
@click.option('--ext', default=('pdf'), show_default=True, type=str, help='Extension to save plot as.')
def metaphlan_report( inp_files, output_dir, ext ):
    '''
    Generate a report on metaphlan3 output profile results
    '''
    if len(inp_files) < 1:
        click.ClickException("At least one input file needs to be provided.")
    save_report(inp_files, output_dir, ext)
    
if __name__ == '__main__':
    cli(obj={})