#!/bin/bash
apt-get install -y unzip
apt-get install -y make
apt-get install -y g++
wget https://github.com/DaehwanKimLab/centrifuge/archive/refs/tags/v1.0.4-beta.zip
unzip v1.0.4-beta.zip
mv /centrifuge-1.0.4-beta /src/
cd /src/centrifuge-1.0.4-beta; make
echo 'export PATH="${PATH}:/src/centrifuge-1.0.4-beta"' >> /root/.bashrc
cd /databases/
/src/centrifuge-1.0.4-beta/centrifuge-download -o taxonomy taxonomy
/src/centrifuge-1.0.4-beta/centrifuge-download -o library -m -d "archaea,bacteria,viral" refseq > seqid2taxid.map
cat library/*/*.fna > input-sequences.fna
centrifuge-build -p 4 --conversion-table seqid2taxid.map \
                 --taxonomy-tree taxonomy/nodes.dmp --name-table taxonomy/names.dmp \
                 input-sequences.fna abv 
rm -rf library
rm -rf taxonomy
