FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y wget
RUN mkdir /src
COPY src/setup.sh /src/
RUN chmod +x /src/setup.sh
RUN /src/setup.sh
