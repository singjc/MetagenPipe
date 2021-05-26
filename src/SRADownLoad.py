from bs4 import BeautifulSoup
import requests
import os
import pandas as pd
import argparse
import re
import gc

### Functions to download SRA data given experiment accessions
### Can be used as a command line tool or


def QuerySRA(expt_acc):
    """
    Query an Experiment Accession to get SampleID and RunID via html scraping
    :param expt_acc: experiment accession
    :return: dictionary containing SampleID (list), RunID (list)

    Note: This function relies on the assumption that, given an SRA experiment query,
    the html document returned will be in a regular format. This function will check for
    1 instance of a sra-full-data class tag (CSS class for div tag) with Sample: in its contents,
    and for 1 instance of a sra-full-data class tag with Runs: in its contents
    """

    # check that accession number in correct format
    if not re.match('^[A-Z]{3}[0-9]+', expt_acc):
        raise ValueError('invalid experiment accession ' + expt_acc)
    # setup dict for output
    return_dict = {'ExptAcc': [], 'Alias': [], 'RunID': []}
    # run http get request
    esearch_req = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=sra&term=' + expt_acc)
    esearch_soup = BeautifulSoup(esearch_req.text, "xml")
    esearch_ids = esearch_soup.find_all('Id')
    if not len(esearch_ids) == 1:
        raise ValueError('unhandled response, expect only 1 ID associated with accession')
    esearch_id_use = esearch_soup.Id.contents[0]
    print(esearch_id_use)
    efetch_req = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=' + esearch_id_use)
    efetch_soup = BeautifulSoup(efetch_req.text, "xml")
    # get runs
    run_set = efetch_soup.RUN_SET
    run_tags = run_set.find_all('RUN')
    if len(run_tags) < 1:
        raise ValueError('expect 1 or more runs associated with accession')
    for run in run_tags:
       return_dict['RunID'].append(run['accession'])

    # get sample id (alias)
    sample_attrs = efetch_soup.SAMPLE.find_all('SAMPLE_ATTRIBUTE')
    alias = None
    for attr in sample_attrs:
        tag_name = attr.TAG.contents[0]
        if tag_name == 'Alias':
            if not alias is None:
                raise Exception('encountered more than one ALIAS under sample attributes')
            alias = attr.VALUE.contents[0]

    if alias is None:
        raise Exception('expected alias to be present in sample attributes')

    for i in range(0, len(return_dict['RunID'])):
        # append one instance of alias, expt accession for each run
        return_dict['Alias'].append(alias)
        return_dict['ExptAcc'].append(expt_acc)

    return return_dict


def DownloadRun(run_acc, download_dir):
    """
    Download a run given a run accession
    :param run_acc: run accession
    :return: makes system call to sra toolkit prefetch command
    """
    cwd = os.getcwd()
    try:
        if not os.path.exists(download_dir):
            print('Making new directory ' + download_dir)
            os.mkdir(download_dir)
        os.chdir(download_dir)
        code0 = os.system('prefetch ' + run_acc)
        code1 = os.system('fastq-dump ' + run_acc)
        rm_code = os.system('rm -rf ' + run_acc)
        assert code0 == 0
        assert code1 == 0
        assert rm_code == 0
    except:
        msg = 'prefetch exited with code ' + str(code0)
        msg = msg + '; fastq-dump exited with code ' + str(code1)
        msg = msg + '; rm -rf exited with code ' + str(rm_code)
        raise Exception(msg)
    finally:
        os.chdir(cwd)


def RunAll(expt_acc_list, download_dir, overwrite=False):
    """

    :param expt_acc_list: list of experiment accessions
    :param download_dir: directory to store files
    :param overwrite: overwrite previous fastq file if exists
    :return:
    """
    m = 0
    for expt_acc in expt_acc_list:
        print(expt_acc)
        run_dict = QuerySRA(expt_acc)
        run_df = pd.DataFrame(run_dict)
        if m > 0:
            all_runs_df = all_runs_df.append(run_df.copy(deep=True))
        else:
            all_runs_df = run_df.copy(deep=True)
            m += 1
        for run_acc in run_dict['RunID']:
            do_download = True
            fastq_file = run_acc + '.fastq'
            fastq_path = os.path.join(download_dir, fastq_file)
            if os.path.exists(fastq_path):
                if overwrite:
                    try:
                        rm_fastq_cmd = 'rm ' + fastq_path
                        rm_fastq_code = os.system(rm_fastq_cmd)
                        assert rm_fastq_code == 0
                    except:
                        raise ValueError(rm_fastq_cmd + ' had exit status ' + str(rm_fastq_code))
                else:
                    do_download = False

            if do_download:
                DownloadRun(run_acc, download_dir)

        del run_df
        del run_dict
        gc.collect()

    all_runs_df.to_csv(os.path.join(download_dir, 'all_runs.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-E", "--all_expt_accs", nargs="*")
    parser.add_argument("-d", "--download_dir", nargs=1, type=str)
    parser.add_argument("-X", "--overwrite", nargs=1, type=bool)
    args = parser.parse_args()
    all_expt_accs = args.all_expt_accs
    download_dir = args.download_dir[0]
    overwrite = args.overwrite[0]
    RunAll(all_expt_accs, download_dir, overwrite)
