#!/bin/bash -l

# These should be customized based on your permissions
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --job-name="geo-coverage"

# Change this to your email address
#SBATCH --mail-user=you@buffalo.edu
#SBATCH --mail-type=all

module load gcc python

# Make sure this venv exists and has all requirements installed
source venv/bin/activate

# Set the scenario and antenna pattern to match your desired input configuration
scenario="fruit-belt"
pattern="AW2032"
python gc.py -s $scenario -j64 -p $pattern
