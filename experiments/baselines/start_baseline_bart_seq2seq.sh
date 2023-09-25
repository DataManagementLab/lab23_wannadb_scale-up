#!/bin/bash
# 
#SBATCH --job-name=ASET-bart-seq2seq
#SBATCH --output=dgx_run_test.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

config_file=$1

. aset-dgx-venv/bin/activate
#. aset-env/bin/activate

python3 -u ./experiments/baseline_bart_seq2seq.py -c $config_file

