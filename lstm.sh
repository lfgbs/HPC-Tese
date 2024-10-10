#!/bin/bash

#SBATCH -p hpc
#SBATCH --ntasks=4

module load udocker/1.3.16

udocker pull python:3.12-slim
udocker create --name=python-models python:3.12-slim
udocker run -v ~/udocker-test:/app python-models python3 /app/Pipeline_LSTM.py