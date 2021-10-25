import sys
from Bio import Entrez

def get_tax_id(species):
    """
    to get data from ncbi taxomomy, we need to have the taxid. we can
    get that by passing the species name to esearch, which will return
    the tax id

    :param species: (string) species name to get tax id for
    :return: (string) returns taxid
    """
    species = species.replace(' ', "+").strip()
    Entrez.email = ""
    search = Entrez.esearch(term = species, db = "taxonomy", retmode = "xml")
    record = Entrez.read(search)
    return record['IdList'][0] if len(record['IdList'])>0 else None

def get_tax_data(taxid):
    """
    Fetch the record for given tax id

    :param taxid: (string) taxid to query ncbi
    :return: (dict) returns a dictionary with information for given tax id
    """
    Entrez.email = ""
    search = Entrez.efetch(id = taxid, db = "taxonomy", retmode = "xml")
    return Entrez.read(search)

def get_tax_rank(species, rank="phylum"):
    """
    Fetch information for a given species and return the rank of the species

    :param species: (string) species name to get tax id for
    :param rank: (string) the rank to return. Default: 'phylum'
    :return: (string) returns the rank of the species
    """
    tax_id = get_tax_id(species)
    print(species)
    if tax_id is None:
        return "Unknown"
    tax_info = get_tax_data( tax_id )[0]
    rank_dict = next((item for item in tax_info["LineageEx"] if item["Rank"] == rank), None)
    return rank_dict['ScientificName'] if rank_dict is not None else "Unknown"


