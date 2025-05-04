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

#SBATCH --mail-user=$1
#SBATCH --mail-type=all

module load gcc python

# Make sure this venv exists and has all requirements installed
source venv/bin/activate

# Set the scenario and antenna pattern to match your desired input configuration
scenario=$2
pattern=$3
python gc.py -s $scenario -j64 -p $pattern
