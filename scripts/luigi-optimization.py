import itertools
import geopandas as gpd
import antenna
import surface_profile
import path_loss

from shapely.geometry import Point
from antenna import Antenna

# Define the lat/lon for each BS
bs_positions = [
    Point(-78.849237, 42.902081),
    Point(-78.864652, 42.901791),
    Point(-78.857450, 42.899530)
]

# Define a list of antenna azimuths for all BS
antenna_azimuths = [[44, 117], [81, 57], [210, 19]]

# Load pattern files
pattern_h = antenna.load_pattern('patterns/horizontal.pat')
pattern_v = antenna.load_pattern('patterns/vertical.pat')

# Load LiDAR point clouds and create ITM model
gdf = gpd.read_file('scenarios/fruit-belt.geojson')
lpcs = surface_profile.load_all_lpcs(gdf, 'lidar')
model = path_loss.ItmModel(lpcs, 90)

# Create all antennas leaving the channel (f_mhz) unset
antennas = []
for i, pos in enumerate(bs_positions):
    for azimuth in antenna_azimuths[i]:
        antennas.append(Antenna(0, 30, pos, 30, azimuth, 1, pattern_h, pattern_v, model))

# Define a list of coverage points
coverage_points = [
    Point(-78.866630, 42.902516),
    Point(-78.861736, 42.897587),
    Point(-78.850725, 42.899603),
    Point(-78.861201, 42.897923),
    Point(-78.856651, 42.902096),
    Point(-78.852102, 42.899211),
    Point(-78.855045, 42.898287),
    Point(-78.852254, 42.902600),
    Point(-78.857607, 42.900415),
    Point(-78.865865, 42.901732)
]

# Define a list of channels/frequencies
channels = [3560, 3590]

# Get index permutations (antenna, channel, point) for PL and (antenna, point) for gain
path_loss_permutations = list(itertools.product(
    list(range(len(antennas))),
    list(range(len(channels))),
    list(range(len(coverage_points)))
))
gain_permutations = list(itertools.product(
    list(range(len(antennas))),
    list(range(len(coverage_points)))
))

# Generate path loss CSV
"""
print('Antenna index, Channel index, Point index, Path loss dB')
for config in path_loss_permutations:
    antenna = antennas[config[0]]
    channel = channels[config[1]]
    point = coverage_points[config[2]]
    antenna.f_mhz = channel
    path_loss = antenna.pathloss(point, 1)
    print(config[0], config[1], config[2], path_loss, sep=', ')
"""

# Generate gain CSV
print('Antenna index, Point index, Gain dB')
for config in gain_permutations:
    antenna = antennas[config[0]]
    point = coverage_points[config[1]]
    gain = antenna.power(point, 1) - antenna.tx_power
    print(config[0], config[1], gain, sep=', ')
    
