from bs4 import BeautifulSoup
import bs4
import requests
import os
import pandas as pd
import argparse
import re

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
    if not re.match('^(ERS|SRS)[0-9]+'):
        raise ValueError('invalid experiment accession ' + expt_acc)
    # run http get request
    sra_req = requests.get('https://www.ncbi.nlm.nih.gov/sra/' + expt_acc)
    sra_soup = BeautifulSoup(sra_req.text)
    return_dict = {'SampleID': [], 'RunID': []}
    sample_id = None
    found_run = False

    try:
        tag_list = sra_soup.find_all(class_='sra-full-data')
        for tag in tag_list:
            contents = tag.contents
            cont0 = contents[0]
            if not type(cont0) == 'str':
                raise ValueError('Expect String in first content entry of sra-full-data tag')
            is_sample = cont0 == 'Sample: '
            is_run = cont0 == 'Run :'
            if not is_sample or is_run:
                next
            span_tag = tag.find_all('span')[0]

            if is_sample:
                if not sample_id is None:
                    raise ValueError('expect to only encounter sample div tag (sra-full-data tag) once')
                # you should only have this if statement run once and only once
                sample_id = span_tag.contents[0]
            elif is_run:
                if found_run:
                    raise ValueError('expect to only encounter run div tag (sra-full-data tag) once')
                found_run = True
                # expect all a tags (hyperlinks) in the span tag will be hyperlinks to runs, which in turn contain
                # run ids
                a_tags = span_tag.find_all('a')
                for a in a_tags:
                    # extract run id from href
                    href = a['href']
                    if not isinstance(href, str):
                        raise ValueError('href in a tag must be string')
                    run_acc_list = re.findall('[A-Z]{3}[0-9]+$', href)
                    # must have 1 and only 1 run accession
                    if len(run_acc_list) > 1:
                        raise ValueError('multiple matches for regex in href')
                    elif len(run_acc_list) < 1:
                        raise ValueError('no matches for regex in href')
                    run_acc = run_acc_list[0]
                    return_dict['RunID'].append(run_acc)
            else:
                raise Exception('did not next() to next iteration in loop and is not a Sample or Run div tag')

        for i in range(0, len(return_dict['RunID'])):
            # append one instance of sample id for each run
            return_dict['SampleID'].append(sample_id)

    except:
        raise Exception('Failure in parsing returned html. Likely unexpected format or bad request')

    return return_dict


def DownloadRun(run_acc, download_dir):
    """
    Download a run given a run accession
    :param run_acc: run accession
    :return: makes system call to sra toolkit prefetch command
    """
    try:
        cwd = os.getcwd()
        os.chdir(download_dir)
        code0 = os.system('prefetch ' + run_acc)
        code1 = os.system('fastq-dump ' + run_acc)
    except:
        msg = 'prefetch exited with code ' + code0
        msg = msg + '; fastq-dump exited with code ' + code1
        raise Exception(msg)


def RunAll(expt_acc_list, download_dir):
    """

    :param expt_acc_list: list of experiment accessions
    :param download_dir: directory to store files
    :return:
    """
    m = 0
    for expt_acc in expt_acc_list:
        run_dict = QuerySRA(expt_acc)
        run_df = pd.DataFrame(run_dict)
        if m > 0:
            all_runs_df.append(run_df)
        else:
            all_runs_df = run_df
        m += 1
        for run_acc in run_dict['RunID']:
            DownloadRun(run_acc, download_dir)

    all_runs_df.to_csv(os.path.join(download_dir, all_runs_df))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-E", "--all_expt_accs")
    parser.add_argument("-d", "--download_dir")
    args = parser.parse_args()
    all_expt_accs = args.all_expt_accs
    download_dir = args.download_dir
    RunAll(all_expt_accs, download_dir)
