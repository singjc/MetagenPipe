try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np
import pandas as pd
import os
import click

# in_files = "/project/workflow/results/metaphlan_profiles/"
# files_path = "/media/justincsing/ExtraDrive1/Documents2/Roest_Lab/Github/microbiome_OJS/workflow/results/metaphlan_profiles/"
# in_files = [ os.path.join(files_path, f) for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f)) ]

def save_report( in_files, out_path=os.getcwd(), ext="pdf" ):
    '''
    Generate a report on metaphlan3 output profile results
    :param in_files: list or iterable containing files to be processed
    :param out_path: str - Path to save plot. (Default: cwd)
    :param ext: str - Extension to save plot as. (Default: pdf)
    :return: None
    '''
    click.echo( f"INFO: There are {len(in_files)} metaphlan profile input files.")
    ## Read in files to a dataframe
    list_dfs = []
    for file in in_files:
        df = pd.read_csv( file, sep="\t", index_col=None, header=0, skiprows=3)
        df['file'] = os.path.basename(file)
        list_dfs.append( df )
    ## Join all dataframes into one
    master_df = pd.concat( list_dfs )
    ## Initialize figure plotting 
    plt.close("all")
    fig = plt.figure(figsize=(8,10))
    ## Plot the distribution of the number of identified clades per fastq file
    count_df = master_df[['#clade_name', 'file']].groupby('file').agg(['count'])
    count_df.columns = count_df.columns.droplevel()
    count_df.rename( columns={'count':'Number_of_Identifications'}, inplace=True )
    count_df_known = count_df[ (count_df[['Number_of_Identifications']]!=1).to_numpy() ]
    count_df_unknown = count_df[ (count_df[['Number_of_Identifications']]<=1).to_numpy() ]
    ax1 = fig.add_subplot(221)
    count_df_unknown.hist(bins=15, ax=ax1, label="Unknown Identifications", width=2)
    count_df_known.hist(bins=15, ax=ax1, label="Actual Identifications", alpha=0.6)
    ax1.set_title("Distribution of Number of Identifications per File")
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("Number of Identifications per File")
    ax1.legend()
    ## Plot distribution of relative abundance
    df_known = master_df[['relative_abundance']][ (master_df[['#clade_name']]!="UNKNOWN").to_numpy() ]
    df_unknown = master_df[['relative_abundance']][ (master_df[['#clade_name']]=="UNKNOWN").to_numpy() ]
    ax2 = fig.add_subplot(222)
    df_unknown.hist(bins=15, ax=ax2, label="Unknown Identifications", width=2)
    df_known.hist(bins=15, ax=ax2, label="Actual Identifications", alpha=0.6)
    ax2.set_title("Distribution of Relative Abundance")
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel("Relative Abundance")
    ax2.legend()
    ## Plot distribution of the number of clade groups
    count_df = master_df[['#clade_name', 'file']].groupby('#clade_name').agg(['count'])
    count_df.columns = count_df.columns.droplevel()
    count_df.rename( columns={'count':'Number_of_Clades'}, inplace=True )
    df_known = count_df[['Number_of_Clades']][ (count_df.index!="UNKNOWN") ]
    df_unknown = count_df[['Number_of_Clades']][ (count_df.index=="UNKNOWN") ]
    ax3 = fig.add_subplot(223)
    df_unknown.hist(bins=15, ax=ax3, label="Unknown Identifications", width=2)
    df_known.hist(bins=15, ax=ax3, label="Actual Identifications", alpha=0.6)
    ax3.set_title("Number of Clade Groups")
    ax3.set_ylabel("Frequency")
    ax3.set_xlabel("Number of Clade Groups")
    ax3.legend()
    ## Save Figure
    outfilepath = out_path + "/" + "metaphlan_profile_report" + "." + ext
    click.echo(f"INFO: Saving report to {outfilepath}")
    plt.tight_layout()
    plt.savefig( outfilepath )