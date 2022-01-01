#!/bin/bash 
apt-get install rsync
kraken2-build --standard --db ./kraken2_db --threads 24
