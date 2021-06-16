FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get install -y less
RUN apt-get install -y graphviz
RUN apt-get install -y git
RUN touch /root/.bashrc
RUN mkdir /src
RUN mkdir /misc_files
COPY jdk-15.0.2_linux-x64_bin.deb /misc_files/
RUN dpkg -i /misc_files/jdk-15.0.2_linux-x64_bin.deb
RUN apt-get install -y unzip                                                                                            
RUN apt-get install -y make                                                                                             
RUN apt-get install -y g++                                                                                              
RUN git clone https://github.com/DerrickWood/kraken2
RUN mv kraken2 /src/
ENV PATH="$PATH:/src/kraken2"
RUN cd /src/kraken2/; install_kraken2.sh ./
RUN rm -rf /src/setup.sh
COPY bin/setup.sh /src/
RUN chmod +x /src/setup.sh
RUN /src/setup.sh

