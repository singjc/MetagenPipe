#!/bin/usr/python
import pandas as pd
import numpy as np
import click
import sys
import os
import subprocess
import multiprocessing
from datetime import datetime


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

@click.group(chain=True)
@click.version_option()
def cli():
    '''
    Run seqtk subsampling
    '''

# Main 
@cli.command()
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--subsample_fraction', default=50000, show_default=True, type=float, help='Data fraction used for subsampling.')
@click.option('--two_pass_mode/--no-two_pass_mode', default=False, show_default=True, help='Enable 2 pass mode')
@click.option('--rng_seed', default=100, show_default=True, type=float, help='Random seed, remember to use the same random seed to keep pairing.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
def main( fastq_files, subsample_fraction, two_pass_mode, rng_seed, nthreads ):
    '''
    Main function call to subsample fastq files using seqtk
    '''
    
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")

    #fastq_file="/project/data/raw/SRA/ERR2017814_50k_subsample.fastq"
    
    fastq_files = list(fastq_files)
    subsample_fraction_pool = [subsample_fraction] * len(fastq_files)
    two_pass_mode_pool = [two_pass_mode] * len(fastq_files)
    rng_seed_pool = [rng_seed] * len(fastq_files)
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( seqtk_call, zip(fastq_files, subsample_fraction_pool, two_pass_mode_pool, rng_seed_pool) )
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
