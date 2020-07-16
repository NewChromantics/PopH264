#!/bin/bash -e

sudo apt-get update
sudo apt-get install libx264-dev gcc g++ -y

g++ -v

ls

make -f $MAKEFILE GithubWorkflow -C PopH264.Linux/