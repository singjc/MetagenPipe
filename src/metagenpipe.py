import pandas as pd
import numpy as np
import click
import sys
import os
import glob
import multiprocessing
import re
from datetime import datetime

from preprocessing.data_processing import seqtk_call, kneaddata_call, metaphlan_call, parse_metaphlan_file, kraken2_call, parse_kraken2_freq
from preprocessing.report import save_report
from warnings import warn
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
@click.option('--overwrite', default=False, show_default=True, type=bool, help='overwrite existing fastq files')
@click.option('--skip', default=True, show_default=True, type=bool, help='skip download upon failure')
@click.option('--accs_only', default=False, show_default=True, type=bool, help='only download accessions and mapped identifiers, no fastqs')
@click.option('--extra_args', type=click.STRING, help='Extra arguments to pass to fastq_dump, encapsulated as a string. i.e. \"-I --gzip --split-files\"')
def sra_downloader( all_expt_accs, download_dir, overwrite, skip, accs_only, extra_args ):
    '''
    Main function call to SRA Downloader
    '''
    RunAll(all_expt_accs, download_dir, overwrite, skip, accs_only, extra_args)

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

def handle_paired_end( fastq_files ):
    """
    Divide paired end fastq files into two lists
    :param fastq_files: fastq files.
    :return: two lists of fastq files, first corresponding to first mate, second corresponding to second mate, and list of file prefixes
    """
    fastq_files = list(set(fastq_files))
    pe_regex = r'_([12]).fastq(.gz)*$'
    r1_regex = r'_(1).fastq(.gz)*$'
    r2_regex = r'_(2).fastq(.gz)*$'
    file_names_prefix = list(set([re.sub(pe_regex, '', os.path.basename(file)) for file in list(fastq_files)]))
    tmp_fastq_files_pair_1 = []
    tmp_fastq_files_pair_2 = []
    for file_prefix in file_names_prefix:
        file_pair = list(set([file for file in list(fastq_files) if file_prefix in file]))
        r1_matches = [file for file in list(file_pair) if re.search(r1_regex, file) is not None]
        r2_matches = [file for file in list(file_pair) if re.search(r2_regex, file) is not None]
        try:
            # assert that each file pair is actually a pair
            assert len(file_pair) == 2
        except:
            raise Exception('for file_prefix {} there are more or less than 2 associated files: {}'.format(file_prefix, file_pair))

        try:
            assert len(r1_matches) == 1
            assert len(r2_matches) == 1
        except:
            raise Exception('for file_prefix {} expect 1 match for read1 and read2. got {} (read1) and {} (read2)'.format(file_prefix, len(r1_matches), len(r2_matches)))

        tmp_fastq_files_pair_1.append(r1_matches[0])
        tmp_fastq_files_pair_2.append(r2_matches[0])
    # print(f"1: {tmp_fastq_files_pair_1}\n2: {tmp_fastq_files_pair_2}")
    # preserve number of fastq files before reassignment
    no_fastq = len(fastq_files)
    try:
        # total number of fastq files should be 2x any of the read pairs
        assert no_fastq == 2 * len(tmp_fastq_files_pair_2)
        assert no_fastq == 2 * len(tmp_fastq_files_pair_1)
    except:
        raise Exception('Not all fastq files have mate pair OR not all fastq files represented in determined paired files - pair1: {} and pair2: {}' .format(tmp_fastq_files_pair_1, tmp_fastq_files_pair_2))

    return tmp_fastq_files_pair_1, tmp_fastq_files_pair_2, file_names_prefix


# Main Data Kneading with kneaddata
@cli.command(name='run-kneaddata', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True
))
@click.argument('fastq_files', nargs=-1, type=click.Path(exists=True))
@click.option('--reference_db', type=click.Path(exists=True), help='Reference Database file.')
@click.option('--output_dir', default=(os.getcwd()+"/kneadeddata/"), show_default=True, type=str, help='Directory to store results.')
@click.option('--trimmomatic', default=(glob.glob("/src/anaconda/envs/**/trimmomatic-*/", recursive=True)[0]), show_default=True, type=str, help='Directory to store results.')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
@click.option('--remove_untarred_fastq/--no-remove_untarred_fastq', default=True, show_default=True, help='Remove untarred fastq file, if a tarred fastq file was used.')
@click.option('--extra_args', type=click.STRING, help='Extra arguments to pass to kneaddata, encapsulated as a string. i.e. \"-q=phred64 --trimmomatic-options=SLIDINGWINDOW:4:20 --trimmomatic-options=MINLEN:50\"')
@click.option('--paired_end/--no-paired_end', default=False, show_default=True, help='Paired end data?')
@click.pass_context
def run_kneaddata( ctx, fastq_files, reference_db, output_dir, trimmomatic, nthreads, remove_untarred_fastq, extra_args, paired_end ):
    '''
    Main function call to process the data with kneaddata
    '''
    if extra_args is not None:
        extra_args = extra_args.split(" ")
    if len(fastq_files) < 1:
        raise click.ClickException("At least one fastq file needs to be provided.")
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    # Check to see if data is paired end
    if paired_end:
        click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Paired end mode selected, pairing fastq files..." )
        # below function splits fastq files by mate
        tmp_fastq_files_pair_1, tmp_fastq_files_pair_2, _ = handle_paired_end(fastq_files)
        fastq_files = tmp_fastq_files_pair_1

    else:
        tmp_fastq_files_pair_2 = [None] * len(fastq_files)
    # Prepare args for parallel processing
    fastq_files = list(fastq_files)
    reference_db_pool = [reference_db] * len(fastq_files)
    output_dir_pool = [output_dir] * len(fastq_files)
    trimmomatic_pool = [trimmomatic] * len(fastq_files)
    remove_untarred_fastq_pool = [remove_untarred_fastq] * len(fastq_files)
    extra_args_pool = [extra_args] * len(fastq_files)
    paired_end_pool = [paired_end] * len(fastq_files)
    paired_end_2_fastqs_pool = tmp_fastq_files_pair_2
    # Initiate a pool with nthreads for parallel processing
    pool = multiprocessing.Pool( nthreads )
    pool.starmap( kneaddata_call, zip(fastq_files, reference_db_pool, output_dir_pool, trimmomatic_pool, remove_untarred_fastq_pool, extra_args_pool, paired_end_pool, paired_end_2_fastqs_pool) )
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


# Main Kraken2

@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--db_use', default=None, show_default=True, type=str, help='kraken2 database to use')
@click.option('--output_dir', default='./', show_default=True, type=str, help='output directory for files')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
@click.option('--paired_end', default=False, show_default=True, type=bool, help='Paired end fastq input, True or False')
def run_kraken2( inp_files, db_use, output_dir, nthreads, paired_end ):
    '''
    Main function call to process the data with metaphlan
    '''
    if len(inp_files) < 1:
        click.ClickException("At least one input file needs to be provided.")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if paired_end:
        click.echo( f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INFO: Paired end mode selected, pairing fastq files..." )
        # below function splits fastq files by mate
        fastq1_files, fastq2_files, prefixes = handle_paired_end(inp_files)
    else:
        fastq1_files = inp_files
        fastq2_files = [None] * len(inp_files)
        prefixes = [re.sub('\\.[A-z]*$', '', os.path.basename(x)) for x in fastq1_files]

    for i in range(len(fastq1_files)):
        fastq1 = fastq1_files[i]
        fastq2 = fastq2_files[i]
        prefix_i = prefixes[i]
        read_file_output = os.path.join(output_dir, prefix_i + '_read_output.txt')
        freq_file_output = os.path.join(output_dir, prefix_i + '_freq_output.txt')
        exit_code = kraken2_call(fastq1=fastq1, fastq2=fastq2, db_use=db_use, reads_file=read_file_output, freq_file=freq_file_output, nthreads=nthreads)
        if not exit_code == 0:
            warn('kraken2 returned exit code {0} for file(s) fastq1: {1} fastq2: {2}'.format(exit_code, fastq1, str(fastq2)))


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

    m = 0

    for f in inp_files:
        df = parse_metaphlan_file(f)
        if df.shape[0] > 0:
            df['filename'] = np.repeat(os.path.basename(f), df.shape[0])
        else:
            warn('file {} lacks any detected taxa, skipping'.format(f))
            continue

        if m == 0:
            m += 1
            master_df = df
        else:
            master_df = master_df.append(df)

    freq_df = master_df.loc[:, ['freq', 'tax_name', 'filename']]

    click.echo(f"INFO: freq_df shape:{freq_df.shape}")
    print(freq_df.head())
    freq_mat_df = freq_df.pivot(index='filename', columns='tax_name', values='freq')
    freq_mat_df.to_csv(os.path.join(output_dir, outfile))

@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output_dir', default=(os.getcwd()), show_default=True, type=str, help='Directory to store matrix output')
@click.option('--freq_mat_file', default=('relative_abundances.csv'), show_default=True, type=str, help='filename for matrix output, relative abundances')
@click.option('--count_mat_file', default=('assignment_counts.csv'), show_default=True, type=str, help='filename for matrix output, reads assigned to taxa')
@click.option('--tax_level_select', default=('^S$'), show_default=True, type=str, help='taxonomic level to index. should be given as a regex, as there appear to be subdivisions of certain taxonomic levels')
def parse_kraken2_multi( inp_files, output_dir='./', freq_mat_file='relative_abundances.csv', count_mat_file='assignment_counts.csv' ,
                         tax_level_select='^S$'):
    """

    :param inp_files: input files. should be a kraken2 relative abundance report
    :param output_dir: Directory to store output matrices
    :param freq_mat_file: filename for matrix output, relative abundances
    :param count_mat_file: filename for matrix output, reads assigned to taxa
    :param tax_level_select: taxonomic level to index. should be given as a regex, as there appear to be subdivisions of certain taxonomic levels'
    :return:
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Ensure in_files are unique
    ## When running on the cluster, files get passed twice when running paired end workflows
    inp_files = list(set(inp_files))

    m = 0

    for f in inp_files:
        df = parse_kraken2_freq(f)
        tax_level = df['tax_level'].to_numpy().astype('str')
        idx_keep = np.array([re.match(tax_level_select, x) is not None for x in tax_level])
        df = df.iloc[idx_keep, :]
        if df.shape[0] > 0:
            df['filename'] = np.repeat(os.path.basename(f), df.shape[0])
        else:
            warn('file {} lacks any detected taxa, skipping'.format(f))
            continue

        if m == 0:
            m += 1
            master_df = df
        else:
            master_df = master_df.append(df)
    
    freq_df = master_df.loc[:, ['freq', 'tax_name', 'filename']]
    count_df = master_df.loc[:, ['total_assigned', 'tax_name', 'filename']]
    
    click.echo(f"INFO: freq_df shape:{freq_df.shape}")
    print(freq_df.head())
    idx = pd.Index(freq_df.index)
    click.echo(f"INFO: count_df shape:{count_df.shape}")
    print(count_df.head())
    freq_mat_df = freq_df.pivot(index='filename', columns='tax_name', values='freq')
    count_mat_df = count_df.pivot(index='filename', columns='tax_name', values='total_assigned')

    freq_mat_df.to_csv(os.path.join(output_dir, freq_mat_file))
    count_mat_df.to_csv(os.path.join(output_dir, count_mat_file))




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

# Concatenate paired end reads
@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output_dir', default=(os.getcwd()), show_default=True, type=str, help='Directory to store output files')
def concat_reads( inp_files, output_dir='./' ):
    """
    Concatenate paired end reads to single file
    :param inp_files: list of input fastq files
    :param output_dir: output directory
    :return: None.
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fastq1_files, fastq2_files, prefixes = handle_paired_end(inp_files)
    for i in range(len(fastq1_files)):
        fastq1 = fastq1_files[i]
        fastq2 = fastq2_files[i]
        prefix_i = prefixes[i]
        output_file = os.path.join(output_dir, prefix_i + '_concat.fastq')
        os.system('cat {} {} > {}'.format(fastq1, fastq2, output_file))

def cmd_wrapper( cmd ):
    exit_status = os.system(cmd)
    if not exit_status == 0:
        warn('nonzero exit status for command: ' + cmd)
        warn('command exited with status {}'.format(exit_status))

# Run humann3 with the ability to skip upon failure
@cli.command()
@click.argument('inp_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output_dir', default=(os.getcwd()), show_default=True, type=str, help='Directory for output file')
@click.option('--nthreads', default=1, show_default=True, type=int, help='Number of threads to use for parallel processing.')
@click.option('--nucleotide_database', default='/project/data/raw/humann3_db/chocophlan')
@click.option('--protein_database', default='project/data/raw/humann3_db/uniref')
def run_humann3( inp_files,
                 output_dir='./',
                 nthreads=1,
                 nucleotide_database='/project/data/raw/humann3_db/chocophlan',
                 protein_database='/project/data/raw/humann3_db/uniref' ):
    """

    :param inp_files: input fastq files. for paired end data, recommended that you concatenate reads for each file pair into single file with concat_reads
    :param output_dir: output directory
    :param nthreads: number of threads to use
    :param nucleotide_database: path to sequence library for humann3
    :param protein_database: path to protein sequence library for humann3
    :return: None 
    write humann3 output files. see (https://github.com/biobakery/biobakery/wiki/humann3#23-humann-default-outputs)
    for more details
    """
    fastq_regex = '\\.fastq$'
    for fname in inp_files:
        try:
            assert re.search(fastq_regex, fname)
        except:
            raise ValueError('expect fastq input')
        humann_cmd = "humann3 --input {} --output {} --threads {} --nucleotide-database {} --protein-database {}".format(fname, output_dir, nthreads, nucleotide_database, protein_database)
        cmd_wrapper( humann_cmd )
        # renormalize gene family and pathway abundance files
        file_prefix = re.sub(fastq_regex, '', os.path.basename(fname))
        gene_fam_file = os.path.join(output_dir, "{}_genefamilies.tsv".format(file_prefix))
        gene_fam_renorm = os.path.join(output_dir, "{}_genefamilies_cpm.tsv".format(file_prefix))
        renorm_gene_fam_cmd = "humann_renorm_table --input {} --output {} --units cpm".format(gene_fam_file, gene_fam_renorm)
        cmd_wrapper(renorm_gene_fam_cmd)
        path_abundance_file = os.path.join(output_dir, "{}_pathabundance.tsv".format(file_prefix))
        path_abundance_renorm = os.path.join(output_dir, "{}_pathabundance_cpm.tsv".format(file_prefix))
        renorm_path_cmd = "humann_renorm_table --input {} --output {} --units cpm".format(path_abundance_file, path_abundance_renorm)
        cmd_wrapper(renorm_path_cmd)


@cli.command()
@click.argument('inp_file', nargs=1, type=click.Path(exists=True))
@click.option('--output_dir', default=(os.getcwd()), show_default=True, type=str, help='Directory for output file')
@click.option('--output_file', default=('humann3_matrix.tsv'), show_default=True, type=str, help='output file name')
@click.option('--log_transform', default=False, show_default=True, type=bool, help='log transform data')
@click.option('--run_id_regex', default=('^[A-Z]{3}[0-9]+'), show_default=True, type=str, help='run id regex')
@click.option('--file_suffix', default=('_1_fastq.gz'), show_default=True, type=str, help='input file suffix')
def parse_humann3( inp_file,
                   output_dir=os.getcwd(),
                   output_file='humann3_matrix.tsv',
                   log_transform=False,
                   run_id_regex='^[A-Z]{3}[0-9]+',
                   file_suffix='_1_fastq.gz'):
    """
    Parse humann3 matrix
    :param inp_file: input file
    :param output_dir: output directory
    :param output_file: output file name
    :param log_transform: whether to log transform data. for CPM data this would be useful
    :return: Nothing. writes transformed matrix in format suitable for Select_Data_sex_complete.ipynb
    """

    inp_mat = pd.read_csv(inp_file, sep='\t', header=1)
    transp_mat = inp_mat.T

    if log_transform:
        log_transp_mat = pd.DataFrame(data=np.log(transp_mat.to_numpy().astype('float32') + 1.),
                                      index=transp_mat.index,
                                      columns=transp_mat.columns)
        transp_mat = log_transp_mat

    transp_mat['filename'] = np.array([re.findall(run_id_regex, x)[0] + file_suffix for x in inp_mat['filename'].to_numpy().astype('str')])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    transp_mat.to_csv(os.path.join(output_dir, output_file))

if __name__ == '__main__':
    cli(obj={})
