FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get install -y less
RUN touch /root/.bashrc
RUN mkdir /src
RUN mkdir /misc_files
COPY jdk-15.0.2_linux-x64_bin.deb /misc_files/
COPY src/setup.sh /src/
RUN chmod +x /src/setup.sh
RUN /src/setup.sh
RUN echo 'export PATH="$PATH:/root/sratoolkit.2.10.9-ubuntu64/bin"' >> /root/.bashrc

