FROM ubuntu

RUN apt-get -y update && \
    apt-get install g++ make -y

COPY / /build

ENTRYPOINT ["make", "-C", "PopH264.Linux/"]
