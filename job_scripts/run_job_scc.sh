#!/bin/bash -l

# Set SCC project
#$ -P biophys

##$ -l buyin

# Specify job name
#$ -N push_pull
# Join output and error files
#$ -j y

# Set walltime
#$ -l h_rt=168:00:00

# Request 1 CPU
#$ -pe omp 1

# Job Array
#$ -t 1


module load miniconda/4.7.5

source activate default

# export MKL_NUM_THREADS=1

which python

SEED=$(($SGE_TASK_ID-1))   

time python run_fit_push.py