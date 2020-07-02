FROM ubuntu

# Avoid user interaction with tzdata
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get dist-upgrade && \
    apt-get install wget build-essential gawk gcc g++ gfortran git texinfo bison libncurses-dev -y && \
    apt-get clean 

RUN wget -c https://sourceforge.net/projects/raspberry-pi-cross-compilers/files/Raspberry%20Pi%20GCC%20Cross-Compiler%20Toolchains/Buster/GCC%2010.1.0/Raspberry%20Pi%203A%2B%2C%203B%2B%2C%204/cross-gcc-10.1.0-pi_3%2B.tar.gz/download -O - | tar -xz -C /opt
    
ENV PATH=/opt/cross-pi-gcc-10.1.0-2/bin:$PATH 
ENV LD_LIBRARY_PATH=/opt/cross-pi-gcc-10.1.0-2/lib:$LD_LIBRARY_PATH

CMD make /build/PopH264.Linux