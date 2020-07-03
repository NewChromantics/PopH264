FROM ubuntu

RUN apt-get -y update && \
    apt-get install g++ make

COPY / /build

ENTRYPOINT ["make", "-C", "/build/PopH264.Linux/"]