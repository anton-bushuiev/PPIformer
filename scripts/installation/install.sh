#!/bin/bash

# Create env
conda create -n ppiformer_u python==3.9.17 -y
conda activate ppiformer_u

# Torch
pip install torch==1.13.1
pip install cmake
pip install torch_geometric==2.3.1
pip install torch-scatter torch-sparse
pip install pytorch_lightning==2.0.8

# latest Graphein
pip install git+https://github.com/a-r-j/graphein.git@master

# Forked equiformer to get attention coefficients
pip install git+https://github.com/anton-bushuiev/equiformer-pytorch.git
# pip install equiformer-pytorch==0.3.9

# Training dependencies
pip install wandb
pip install hydra-core
pip install -U hydra-submitit-launcher

# Install mutils and PPIRef
pip install git+https://github.com/anton-bushuiev/mutils.git
pip install git+https://github.com/anton-bushuiev/PPIRef.git
# alternative for dev
# pip install -e ../PPIRef
# pip install -e ../mutils

# Install current project
pip install -e .
