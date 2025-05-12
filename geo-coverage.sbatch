#!/bin/bash -l

# These can be customized based on your permissions
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --job-name="geo-coverage"

#SBATCH --mail-user: <PUT YOUR EMAIL HERE>
#SBATCH --mail-type=all

module load gcc python

# Make sure this venv exists and has all requirements installed
source ~/venv/bin/activate
python gc.py -s $1 -j64 -p $2
