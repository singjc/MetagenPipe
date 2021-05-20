#!/bin/usr/python
import pandas as pd
import numpy as np
import click
import sys
import os
import glob
import subprocess
import multiprocessing
from datetime import datetime
from pathlib import Path
import re
import time
import shutil

def check_make_output_dir( output_dir ):
    '''
    Check to see if directory exists, otherwise create it.
    '''
    Path( output_dir ).mkdir( parents=True, exist_ok=True )

def check_external_program_install( external_program ):
    '''
    Check to see if an external program is installed
    '''
    exitcode = subprocess.getstatusoutput( external_program )[0]
    assert( exitcode!=127 ), f"Could not verify {external_program}! Make sure you have the program installed!\nExitcode: {exitcode}"


def time_func(func, msg, *args): 
    '''
    function which prints the wall time it takes to execute the given command
    '''
    start_time = time.time()
    func(*args)
    end_time = time.time()
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: {msg} took this long to run: {end_time-start_time} seconds")

def extract_all( archive ):
    '''
    Unpack an archive in the same directory of the located archive
    '''
    extract_path=os.path.dirname(os.path.realpath(archive))
    shutil.unpack_archive(archive, extract_path)

def seqtk_call( fastq_file, subsample_fraction, output_dir=(os.getcwd()+"/raw_subsampled/"), two_pass_mode=False, rng_seed=100, add_file_tag=False  ):
    '''
    Make a system call to seqtk
    '''
    ## Check to make sure seqtk is installed.
    check_external_program_install( "seqtk" )
    ## Check to see if output directory exists, otherwise create it
    check_make_output_dir( output_dir )
    ## Get base filename
    root, ext = os.path.splitext( fastq_file )
    while ext in ['.gz', '.tar']:
        root, ext = os.path.splitext( root )
    ## Check if file is an archive
    fastq_archive=False
    if ".gz" in fastq_file:
        fastq_archive=True
        time_func( extract_all, f"Unpacking {fastq_file}", fastq_file )
        fastq_file = os.path.dirname(os.path.realpath(fastq_file)) + "/" + os.path.basename(root) + ".fastq"
    ## Generate subsampled filename to write to
    fastq_subsampled_file = os.path.basename(root) + "_seqt.subsampled"
    ## Add file tag denoting subsampled file with x seed and n fraction
    if add_file_tag:
        fastq_subsampled_file + "_seed_" + str(round(rng_seed)) + "_fraction_" + str(round(subsample_fraction))
    fastq_subsampled_file = output_dir + "/" + fastq_subsampled_file  + ".fastq"
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
    if fastq_archive:
        click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Removing untarred cached fastq file: {fastq_file}" )
        os.remove( fastq_file )
def kneaddata_call( fastq_file, reference_db, output_dir, trimmomatic, **kwargs ):
    '''
    Make a system call to kneaddata
    '''
    ## Check to make sure kneaddate is installed.
    check_external_program_install( "kneaddata" )
    ## Check to see if output directory exists, otherwise create it
    check_make_output_dir( output_dir )
    ## Get base filename
    root, ext = os.path.splitext( fastq_file )
    while ext in ['.gz', '.tar']:
        root, ext = os.path.splitext( root )
    ## Check if file is an archive
    fastq_archive=False
    if ".gz" in fastq_file:
        fastq_archive=True
        time_func( extract_all, f"Unpacking {fastq_file}", fastq_file )
        fastq_file = os.path.dirname(os.path.realpath(fastq_file)) + "/" + os.path.basename(root) + ".fastq"
    ## Generate list command
    shell_cmd_list = ['kneaddata', '--input']
    shell_cmd_list.append( fastq_file )
    shell_cmd_list.append( '--trimmomatic' )
    shell_cmd_list.append( trimmomatic )
    shell_cmd_list.append( '--reference-db' )
    shell_cmd_list.append( reference_db )
    shell_cmd_list.append( '--output' )
    shell_cmd_list.append( output_dir )
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Executing kneaddate for {fastq_file}" )
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: kneaddata command: {shell_cmd_list}" )
    process = subprocess.Popen( shell_cmd_list, stdout=subprocess.PIPE )
    # Check process
    while True:
        return_code = process.poll()
        if return_code is not None:
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Process has finished with return code: {return_code}" )
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: kneaddata results written to {output_dir,}" )
            break
    if fastq_archive:
        click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Removing untarred cached fastq file: {fastq_file}" )
        os.remove( fastq_file )

def metaphlan_call( input_file, input_type, output_dir_bowtie, output_dir_profile, nthreads ):
    """
    System call to process files with metaphlan3
    :param input_file: list or iterable containing files to be processed
    :param input_type: str, one of 'fastq', 'bowtie2out'. if bowtie2 out, use input files must be alignemnts from bowtie2
     see metaphlan3 documentation for more details.
    :param output_dir_bowtie: output directory for bowtie2 output
    :param output_dir_profile: output directory for functional profiles
    :param nthreads: number of threads to run
    :return: metaphlan output is made in output directory
    """
    check_external_program_install("metaphlan")
    if not os.path.exists(output_dir_profile):
        os.mkdir(output_dir_profile)
    cmd = ['metaphlan', input_file]
    if input_type == 'bowtie2out':
        suffix = '\\.bowtie2\\.bz2$'

    elif input_type == 'fastq':
        suffix = '\\.fastq$'
        bowtie2_file = re.sub(suffix, '.bowtie2.bz2', os.path.basename(input_file))
        if not os.path.exists(output_dir_bowtie):
            os.mkdir(output_dir_bowtie)
        cmd.append('--bowtie2out')
        bowtie2_fpath = os.path.join(output_dir_bowtie, bowtie2_file)
        cmd.append(bowtie2_fpath)
    else:
        raise ValueError('input_type must be fastq for bowtie2out')

    profile_output_file=re.sub(suffix, '_profile.txt', os.path.basename(input_file))
    cmd.append('--nproc')
    cmd.append( str(nthreads) )
    cmd.append('--input_type')
    cmd.append(input_type)
    cmd.append('-o')
    profile_fpath = os.path.join(output_dir_profile, profile_output_file)
    cmd.append(profile_fpath)
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: metaphlan command: {cmd}" )
    process = subprocess.Popen( cmd, stdout=subprocess.PIPE )
    while True:
        return_code = process.poll()
        if return_code is not None:
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Process has finished with return code: {return_code}" )
            if input_type == 'fastq':
                click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: bowtie2 file written to {bowtie2_fpath}" )
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: profile file written to {profile_fpath}" )
            break

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
