#!/bin/bash
# Download files corresponding to publication: https://doi.org/10.1038/s41467-020-18476-8
# supp table 1, corresponding to ~4000 examples in 'discovery' set
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-18476-8/MediaObjects/41467_2020_18476_MOESM3_ESM.xlsx -O ../data/raw/supp_table1.xlsx
# supp table 4, corresponding to ~700 'independent dataset' examples
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-18476-8/MediaObjects/41467_2020_18476_MOESM6_ESM.xlsx -O ../data/raw/supp_table4.xlsx
