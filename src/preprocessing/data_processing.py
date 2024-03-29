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
import gzip
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

def fastq_file_process_check ( fastq_file ):
    """
    Check if a fastq file is archived/compressed.
    Return path to decompressed fastq file and return logical true for decompression 
    """
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
    return fastq_file, fastq_archive, root

def time_func(func, msg, *args): 
    '''
    function which prints the wall time it takes to execute the given command
    '''
    start_time = time.time()
    func(*args)
    end_time = time.time()
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: {msg} took this long to run: {end_time-start_time} seconds")

def gunzip_something(gzipped_file_name, work_dir):
    "gunzip the given gzipped file"

    # see warning about filename
    filename = os.path.split(gzipped_file_name)[-1]
    filename = re.sub(r"\.gz$", "", filename, flags=re.IGNORECASE)

    with gzip.open(gzipped_file_name, 'rb') as f_in:  # <<========== extraction happens here
        with open(os.path.join(work_dir, filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_all( archive ):
    '''
    Unpack an archive in the same directory of the located archive
    '''
    extract_path=os.path.dirname(os.path.realpath(archive))
    # Get current registered unpacking/archival formats
    registered_formats = shutil.get_unpack_formats()
    # Check to see if plain gunzip is part of registered formats
    if ('gz' not in [format[0] for format in registered_formats]):
        shutil.register_unpack_format('gz', ['.gz', ], gunzip_something)
    shutil.unpack_archive(archive, extract_path)

def seqtk_call( fastq_file, subsample_fraction, output_dir=(os.getcwd()+"/raw_subsampled/"), two_pass_mode=False, rng_seed=100, add_file_tag=False, remove_untarred_fastq=True  ):
    '''
    Make a system call to seqtk
    '''
    ## Check to make sure seqtk is installed.
    check_external_program_install( "seqtk" )
    ## Check to see if output directory exists, otherwise create it
    check_make_output_dir( output_dir )
    ## Check if fastq file is compressed
    fastq_file, fastq_archive, root = fastq_file_process_check ( fastq_file )
    ## Generate subsampled filename to write to
    base_name = os.path.basename(root)
    pe_regex = '_[1-2]$'
    pe_match = re.search(pe_regex, base_name)
    pe_suffix = ''
    if pe_match:
        click.echo(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: File {fastq_file} recognized as paired end, keeping paired end suffix")
        pe_suffix += pe_match.group(0)
        base_name = re.sub(pe_regex, '', base_name)
    fastq_subsampled_file = base_name + "_seqt.subsampled" + pe_suffix
    ## Add file tag denoting subsampled file with x seed and n fraction
    if add_file_tag:
        fastq_subsampled_file + "_seed_" + str(round(rng_seed)) + "_fraction_" + str(round(subsample_fraction))
    fastq_subsampled_file = output_dir + "/" + fastq_subsampled_file  + ".fastq"
    ## Check if subsampled file already exists, if it does skip
    if os.path.exists(fastq_subsampled_file):
        click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] WARN: Subsampled file {fastq_subsampled_file} already exits, skipping..." )
        return
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
    if fastq_archive and remove_untarred_fastq:
        click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Removing untarred cached fastq file: {fastq_file}" )
        os.remove( fastq_file )

def kneaddata_call( fastq_file, reference_db, output_dir, trimmomatic, remove_untarred_fastq=True, extra_args=None, paired_end=False, paired_end_2_fastq=None, **kwargs ):
    '''
    Make a system call to kneaddata
    '''
    ## Check to make sure kneaddate is installed.
    check_external_program_install( "kneaddata" )
    ## Check to see if output directory exists, otherwise create it
    check_make_output_dir( output_dir )
    ## Check if fastq file is compressed
    fastq_file, fastq_archive, root = fastq_file_process_check ( fastq_file )
    ## Generate list command
    shell_cmd_list = ['kneaddata', '--input']
    shell_cmd_list.append( fastq_file )
    if paired_end and paired_end_2_fastq is not None:
        ## Check if fastq file is compressed
        paired_end_2_fastq, fastq_archive, root = fastq_file_process_check ( paired_end_2_fastq )
        shell_cmd_list.append( '--input' )
        shell_cmd_list.append( paired_end_2_fastq )
    shell_cmd_list.append( '--trimmomatic' )
    shell_cmd_list.append( trimmomatic )
    shell_cmd_list.append( '--reference-db' )
    shell_cmd_list.append( reference_db )
    shell_cmd_list.append( '--output' )
    shell_cmd_list.append( output_dir )
    ## Add Extra Args 
    if extra_args is not None:
        for item in extra_args:
            key, value = item.split('=')
            shell_cmd_list.append( key )
            shell_cmd_list.append( value )
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Executing kneaddata for {fastq_file}" )
    click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: kneaddata command: {shell_cmd_list}" )
    process = subprocess.Popen( shell_cmd_list, stdout=subprocess.PIPE )
    # Check process
    while True:
        return_code = process.poll()
        if return_code is not None:
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Process has finished with return code: {return_code}" )
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: kneaddata results written to {output_dir,}" )
            break
    if fastq_archive and remove_untarred_fastq:
        click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Removing untarred cached fastq file: {fastq_file}" )
        os.remove( fastq_file )
        if paired_end:
            click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Removing untarred cached fastq file: {paired_end_2_fastq}" )
            os.remove( paired_end_2_fastq )

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
    df = pd.read_csv(input_file, delimiter='\t', skiprows=4, header=None).iloc[:, np.array([0, 2])]
    # taxa is in 1st col, relative abundance in 3rd. See https://github.com/biobakery/biobakery/wiki/metaphlan3#output-files
    # for more details
    df.columns = ['tax_name', 'freq']
    # filter for only taxa with species
    taxa = df['tax_name'].to_numpy().astype('str')
    species_idx = np.argwhere(np.array([re.search('s__', x) is not None for x in taxa])).flatten()
    # pandas gracefully handles cases where species_idx is of length 0 by
    # returning empty dataframe with appropriate columns
    df = df.iloc[species_idx, :]
    return df

def kraken2_call( fastq1, db_use, reads_file, freq_file, nthreads=1, fastq2=None ):
    """
    System Call to Kraken2
    :param fastq1: fastq file
    :param db_use: kraken2 database
    :param reads_file: file written to in --output flag for kraken2. contains annotated reads
    :param freq_file: file written to in --report flag for kraken2. contains taxa frequencies
    :param nthreads: number of threads to use
    :param fastq2: if None, run in single end mode. if not None, run in paired end mode
    :return: exit status for system call
    """
    if fastq2 is not None:
        cmd = 'kraken2 --paired --db {0} --output {1} --report {2} --threads {3} {4} {5}'.format(db_use, reads_file, freq_file, nthreads, fastq1, fastq2)
    else:
        cmd = 'kraken2 --db {0} --output {1} --report {2} --threads {3} {4}'.format(db_use, reads_file, freq_file, nthreads, fastq1)

    exit_status = os.system(cmd)
    return exit_status

def parse_kraken2_freq( freq_file ):
    """

    :param freq_file:
    :return: data frame with taxa frequency information at the specified taxonomic level
    """

    df = pd.read_csv(freq_file, header=None, delimiter='\t')
    # frequency, total assigned reads, total reads directly assigned, taxonomic level, ncbi id, taxonomic name
    df.columns = ['freq', 'total_assigned', 'total_direct', 'tax_level', 'ncbi_id', 'tax_name']
    return df
