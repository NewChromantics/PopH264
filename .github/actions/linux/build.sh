#!/bin/bash -e

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install libx264-dev gcc-10 g++-10 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10

g++ -v

make -f $MAKEFILE GithubWorkflow -C PopH264.Linux/

ls PopH264.Linux

mkdir -p ./build/PopH264$ARCHITECTURE
mv PopH264.Linux/PopH264$ARCHITECTURE.so PopH264.Linux/PopH264TestApp$ARCHITECTURE ./build/PopH264$ARCHITECTURE