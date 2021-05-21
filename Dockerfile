FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get install -y less
RUN apt-get install -y graphviz
RUN apt-get install git
RUN touch /root/.bashrc
RUN mkdir /src
RUN mkdir /misc_files
COPY jdk-15.0.2_linux-x64_bin.deb /misc_files/
RUN dpkg -i /misc_files/jdk-15.0.2_linux-x64_bin.deb
RUN rm -rf /src/setup.sh
RUN rm -rf /src/setup_databases.sh
COPY bin/setup.sh /src/
RUN chmod +x /src/setup.sh
RUN /src/setup.sh
RUN conda activate microbiome
# metaphlan, humann2, + kneaddata install databases
RUN mkdir /databases
RUN metaphlan --install --index mpa_v30_CHOCOPhlAn_201901 --bowtie2db /databases/metaphlan/
RUN humann_databases --download chocophlan full /databases/humann --update-config yes
RUN kneaddata_database --download human_genome bowtie2 /databases/kneaddata_human_bowtie2
RUN conda deactivate

