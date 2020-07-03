FROM ubuntu

RUN apt-get -y update && \
    apt-get install g++ make -y

COPY / /build

ENTRYPOINT ["make", "-C", "/build/PopH264.Linux/"]
