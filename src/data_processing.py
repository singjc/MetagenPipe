#!/bin/usr/python
import pandas as pd
import numpy as np
import click
import sys
import os
import subprocess
import multiprocessing
from datetime import datetime
import re


def check_seqtk_install():
    '''
    Check to see if seqtk is installed
    '''
    exitcode = subprocess.getstatusoutput("seqtk")[0]
    assert( exitcode==127 ), "Could not verify seqtk! Make sure you have the program installed!"

def seqtk_call( fastq_file, subsample_fraction, two_pass_mode=False, rng_seed=100  ):
    '''
    Make a system call to seqtk
    '''
    ## Check to make sure seqtk is installed.
    check_seqtk_install()
    ## Get base filename
    root, ext = os.path.splitext( fastq_file )
    ## Generate subsampled filename to write to
    fastq_subsampled_file = root + "_seed_" + str(rng_seed) + "_fraction_" + str(subsample_fraction) + ".fastq"
    ## Generate list command
    shell_cmd_list = ['seqtk', 'sample']
    if two_pass_mode:
        shell_cmd_list.append( '-2' )
    shell_cmd_list.append( '-s'+str(rng_seed) )
    shell_cmd_list.append( fastq_file )
    shell_cmd_list.append( str(subsample_fraction) )
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Initiating subsampling for {fastq_file}" )
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: seqtk command: {shell_cmd_list}" )
    with open( fastq_subsampled_file, 'w' ) as stdout_file:
        process = subprocess.Popen( shell_cmd_list, stdout=stdout_file )
    # Check process 
    while True:
        return_code = process.poll()
        if return_code is not None:
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Process has finished with return code: {return_code}" )
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Subsampled file written to {fastq_subsampled_file}" )
            break


def check_kneaddata_install():
    '''
    Check to see if kneaddata is installed
    '''
    exitcode = subprocess.getstatusoutput("kneaddata")[0]
    assert( exitcode==127 ), "Could not verify kneaddata! Make sure you have the program installed!"


def kneaddata_call( fastq_file, reference_db, output_dir, **kwargs ):
    ## Check to make sure kneaddate is installed.
    check_kneaddata_install()
    ## Generate list command
    shell_cmd_list = ['kneaddata', '--input']
    shell_cmd_list.append( fastq_file )
    shell_cmd_list.append( '--trimmomatic' )
    shell_cmd_list.append( '/root/anaconda/envs/microbiome/share/trimmomatic-0.39-1/' )
    shell_cmd_list.append( '--reference-db' )
    shell_cmd_list.append( reference_db )
    shell_cmd_list.append( '--output' )
    shell_cmd_list.append( output_dir )
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Executing kneaddate for {fastq_file}" )
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: kneaddate command: {shell_cmd_list}" )
    process = subprocess.Popen( shell_cmd_list, stdout=subprocess.PIPE )
    # Check process
    while True:
        return_code = process.poll()
        if return_code is not None:
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Process has finished with return code: {return_code}" )
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: kneaddata results written to {output_dir,}" )
            break


@click.group(chain=True)
@click.version_option()
def cli():
    '''
    Data Processing
    '''

# Main Subsampling with seqtk
@cli.command()
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--subsample_fraction', default=50000, show_default=True, type=float, help='Data fraction used for subsampling.')
@click.option('--two_pass_mode/--no-two_pass_mode', default=False, show_default=True, help='Enable 2 pass mode')
@click.option('--rng_seed', default=100, show_default=True, type=float, help='Random seed, remember to use the same random seed to keep pairing.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def run_seqtk( fastq_files, subsample_fraction, two_pass_mode, rng_seed, nthreads ):
    '''
    Main function call to subsample fastq files using seqtk
    '''
    
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")

    fastq_files = list(fastq_files)
    subsample_fraction_pool = [subsample_fraction] * len(fastq_files)
    two_pass_mode_pool = [two_pass_mode] * len(fastq_files)
    rng_seed_pool = [rng_seed] * len(fastq_files)
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( seqtk_call, zip(fastq_files, subsample_fraction_pool, two_pass_mode_pool, rng_seed_pool) )
    pool.close()
    pool.join()

# Main Data Kneading with kneaddata
@cli.command()
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--reference_db', type=click.Path(exists=True), help='Reference Datanase file.')
@click.option('--output_dir', default=(os.getcwd()+"/kneadeddata/"), show_default=True, type=str, help='Directory to store results.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def run_kneaddata( fastq_files, reference_db, output_dir, nthreads ):
    '''
    Main function call to process the data with kneaddata
    '''
    
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    fastq_files = list(fastq_files)
    reference_db_pool = [reference_db] * len(fastq_files)
    output_dir_pool = [output_dir] * len(fastq_files)
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( kneaddata_call, zip(fastq_files, reference_db_pool, output_dir_pool) )
    pool.close()
    pool.join()

# MetaPhlan 3.0
def check_metaphlan_install():
    '''
    Check to see if metaphlan is installed
    '''
    exitcode = subprocess.getstatusoutput("metaphlan")[0]
    assert( exitcode==127 ), "Could not verify metaphlan! Make sure you have the program installed!"

def metaphlan_call( input_file, input_type, output_dir_bowtie, output_dir_profile, nthreads ):
    """
    Function call to process files with metaphlan3
    :param input_file: list or iterable containing files to be processed
    :param input_type: str, one of 'fastq', 'bowtie2out'. if bowtie2 out, use input files must be alignemnts from bowtie2
     see metaphlan3 documentation for more details.
    :param output_dir_bowtie: output directory for bowtie2 output
    :param output_dir_profile: output directory for functional profiles
    :param nthreads: number of threads to run
    :return:
    """
    if not os.path.exists(output_dir_profile):
        os.mkdir(output_dir_profile)

    cmd = ['metaphlan', input_file]
    if input_type == 'bowtie2out':
        suffix = '\\.bowtie2\\.bz2$'

    elif input_type == 'fastq':
        suffix = '\\.fastq$'
        bowtie2_file = re.sub(suffix, '.bowtie2.bz2', input_file)
        if not os.path.exists(output_dir_bowtie):
            os.mkdir(output_dir_bowtie)
        cmd.append('--bowtie2out')
        bowtie2_fpath = os.path.join(output_dir_bowtie, bowtie2_file)
        cmd.append(bowtie2_fpath)
    else:
        raise ValueError('input_type must be fastq for bowtie2out')

    profile_output_file=re.sub(suffix, '_profile.txt', input_file)
    cmd.append('--nproc')
    cmd.append(nthreads)
    cmd.append('--input_type')
    cmd.append(input_type)
    cmd.append('-o')
    profile_fpath = os.path.join(output_dir_profile, profile_output_file)
    cmd.append(profile_fpath)
    process = subprocess.Popen(cmd)
    while True:
        return_code = process.poll()
        if return_code is not None:
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Process has finished with return code: {return_code}" )
            if input_type == 'fastq':
                click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: bowtie2 file written to {bowtie2_fpath}" )
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: profile file written to {profile_fpath}" )
            break


@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output_dir_bowtie', default=(os.getcwd()+"/metaphlan_bowtie2out/"), show_default=True, type=str, help='Directory to store bowtie2 results.')
@click.option('--output_dir_profile', default=(os.getcwd()+"/metaphlan_profiles/"), show_default=True, type=str, help='Directory to store species profile results.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def run_metaphlan( inp_files, input_type, output_dir_bowtie, output_dir_profile, nthreads ):

    if len(inp_files) < 1:
        click.ClickException("At least one input file needs to be provided.")

    for f in inp_files:
        metaphlan_call(f, input_type, output_dir_bowtie, output_dir_profile, nthreads)


def parse_metaphlan_file( input_file ):
    """
    Parse metaphlan output file
    :param input_file: str, filepath to input file
    :return: 1 row pandas dataframe, with species abundances in columns

    Note: expect that basename of file is unique

    """
    input = pd.read_csv(input_file, delimiter='\t', skiprows=4, header=False)
    # taxa is in 1st col, relative abundance in 3rd. See https://github.com/biobakery/biobakery/wiki/metaphlan3#output-files
    # for more details
    taxa = input.iloc[: , 0].to_numpy().astype('str')
    rel_abundance = input.iloc[:, 2].to_numpy().astype('float32')
    species_idx = np.where(np.array([re.search('s__', x) is not None for x in taxa]))[0]
    dict_out = {}
    for i in species_idx:
        taxa_name = taxa[i]
        dict_out[taxa_name] = np.array([rel_abundance[i]]).astype('float32')
    # get rid of file ending for row name for output dataframe
    row_name = re.sub('\\.[A-z0-9]+', '', os.path.basename(input_file))
    pd_df_out = pd.DataFrame(dict_out, index=row_name)
    try:
        # check that relative abundance sums to 100 percent
        total_pct = pd_df_out.to_numpy(dtype='float32').sum()
        assert np.equal(total_pct, 100.0)
    except:
        raise ValueError('total relative abundance ' + total_pct + ' is not 100.0')

    return pd_df_out


@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output_dir', default=(os.getcwd()), show_default=True, type=str, help='Directory to store matrix output')
@click.option('--outfile', default=('relative_abundances.csv'), show_default=True, type=str, help='filename for matrix output')
def parse_metaphlan_multi( inp_files, output_dir='./', outfile='relative_abundances.csv' ):
    """
    Parse a list or iterable of metaphlan output files
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


if __name__ == '__main__':
    cli()
