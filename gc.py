from geo_coverage import *

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', type=str)
parser.add_argument('-j', '--cores', type=int)
parser.add_argument('-p', '--pattern', type=str)
parser.add_argument('-g', '--granularity', default=9, type=float)
parser.add_argument('-pg', '--profile-granularity', default=2, type=float)
input = parser.parse_args()

scenario = input.scenario
pattern = input.pattern
granularity = input.granularity
prof_granularity = input.profile_granularity
num_cores = input.cores
map, antennas, gdf = run_analysis(scenario, pattern, granularity, prof_granularity, num_cores)

output_path = scenario + '.png'
print('--- Plotting heatmap to {output_path}...')
plot_coverage_map(map, gdf.iloc[0]['geometry'], antennas, output_path)
print('--- Done.')
