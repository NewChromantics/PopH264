#!/bin/bash -e

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install libx264-dev gcc-10 g++-10 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1

g++ -v

ls

make -f $MAKEFILE GithubWorkflow -C PopH264.Linux/