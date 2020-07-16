#!/bin/bash -e

sudo apt-get update
sudo apt-get install libx264-dev -y

g++ -v

ls

# make -f $MAKEFILE GithubWorkflow -C $GITHUB_WORKSPACE/PopH264.Linux/