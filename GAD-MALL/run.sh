#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
python -u 3D_CNN.py
python -u 3D_CAE.py
python -u Active_learning.py

 
