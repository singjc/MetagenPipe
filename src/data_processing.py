#!/bin/usr/python
import pandas as pd
import numpy as np
import click
import sys
import os
import subprocess
import multiprocessing
from datetime import datetime
from pathlib import Path

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

def kneaddata_call( fastq_file, reference_db, output_dir, **kwargs ):
    '''
    Make a system call to kneaddata
    '''
    ## Check to make sure kneaddate is installed.
    check_external_program_install( "kneaddata" )
    ## Check to see if output directory exists, otherwise create it
    check_make_output_dir( output_dir )
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

def metaphlan_call( **kwargs ):
    '''
    Make a system call to metaphlan
    '''

def prepare_export():
    '''
    Parse output from metaphlan and export an abundance matrix
    '''

# Main Command Line Interface
@click.group(chain=True)
@click.version_option()
@click.pass_context
def cli( ctx ):
    '''
    Raw Data Processing for Microbiome Metagenomics Data
    '''

# Main Subsampling with seqtk
@cli.command()
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--subsample_fraction', default=50000, show_default=True, type=float, help='Data fraction used for subsampling.')
@click.option('--output_dir', default=(os.getcwd()+"/raw_subsampled/"), show_default=True, type=str, help='Directory to store results.')
@click.option('--two_pass_mode/--no-two_pass_mode', default=False, show_default=True, help='Enable 2 pass mode')
@click.option('--rng_seed', default=100, show_default=True, type=float, help='Random seed, remember to use the same random seed to keep pairing.')
@click.option('--add_file_tag/--no-add_file_tag', default=False, show_default=True, help='Add a tag to subsampled files, denoting subsampled file using x seed at n fraction.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def run_seqtk( fastq_files, subsample_fraction, output_dir, two_pass_mode, rng_seed, add_file_tag, nthreads ):
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
    # Initiate a pool with nthreads for parallel processing
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( seqtk_call, zip(fastq_files, subsample_fraction_pool, output_dir_pool, two_pass_mode_pool, rng_seed_pool, add_file_tag_pool) )
    pool.close()
    pool.join()

# Main Data Kneading with kneaddata
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--reference_db', type=click.Path(exists=True), help='Reference Datanase file.')
@click.option('--output_dir', default=(os.getcwd()+"/kneadeddata/"), show_default=True, type=str, help='Directory to store results.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
@click.pass_context
def run_kneaddata( ctx, fastq_files, reference_db, output_dir, nthreads ):
    '''
    Main function call to process the data with kneaddata
    '''
    # Handle additional args TODO: Still not working... extra args are not being passed to the sub command for some reason
    extra_args = dict([item.strip('--').split('=') for item in ctx.args])
    # print( extra_args )
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    # Prepare args for parallel processing
    fastq_files = list(fastq_files)
    reference_db_pool = [reference_db] * len(fastq_files)
    output_dir_pool = [output_dir] * len(fastq_files)
    # Initiate a pool with nthreads for parallel processing
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( kneaddata_call, zip(fastq_files, reference_db_pool, output_dir_pool) )
    pool.close()
    pool.join()

# Main Processing with MetaPhlan
@cli.command()
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def run_metaphlan( fastq_files, nthreads ):
    '''
    Main function call to process the data with metaphlan
    '''
    
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")
    
    # Prepare args for parallel processing
    fastq_files = list(fastq_files)
    # Initiate a pool with nthreads for parallel processing
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( metaphlan_call, zip(fastq_files) )
    pool.close()
    pool.join()

# Main Exporting an Abundance Matrix
@cli.command()
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def export( fastq_files, nthreads ):
    '''
    Main function call to export an abundance matrix
    '''
    
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")
    
    # Prepare args for parallel processing
    fastq_files = list(fastq_files)
    # Initiate a pool with nthreads for parallel processing
    # NOTE: You may not need to use parallel processing to generate the export abundance matrix, depending on how you script it
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( prepare_export, zip(fastq_files) )
    pool.close()
    pool.join() 


if __name__ == '__main__':
    cli()