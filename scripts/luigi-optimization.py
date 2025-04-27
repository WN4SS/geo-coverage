import random
import geopandas as gpd
import antenna
import surface_profile
import path_loss

from shapely.geometry import Point
from antenna import Antenna
from path_loss import FreeSpaceModel

# Load antenna patterns
pattern_h = antenna.load_pattern('patterns/horizontal.pat')
pattern_v = antenna.load_pattern('patterns/vertical.pat')

# Load scenario GeoJSON
gdf = gpd.read_file('scenarios/fruit-belt.geojson')

# Create the path loss model
model = FreeSpaceModel()

def gen_random_points(num):
    """
    Generate a list of random points within the coverage area
    """
    bounding_poly = gdf.geometry.iloc[0]
    lon_min, lat_min, lon_max, lat_max = bounding_poly.bounds
    points = []
    for _ in range(num):
        while True:
            lat = lat_min + random.random() * (lat_max - lat_min)
            lon = lon_min + random.random() * (lon_max - lon_min)
            point = Point(lon, lat)
            if bounding_poly.contains(point):
                points.append(point)
                break
    return points

class BaseStation:
    def __init__(self, num_antennas, channels):
        """
        Create a base station with a specified number of antennas, and channels
        """
        self.channels = channels
        self.antennas = []
        pos = gen_random_points(1)[0]
        for _ in range(num_antennas):
            azimuth = random.random() * 360
            self.antennas.append(Antenna(0, 30, pos, 30, azimuth, 1, pattern_h, pattern_v, model))

    def get_rsrps(self, points):
        """
        Estimate the RSRP for every permutation of channel, antenna, and coverage point
        """
        antenna_rsrps = []
        for antenna in self.antennas:
            channel_rsrps = []
            for channel in self.channels:
                point_rsrps = []
                for point in points:
                    antenna.f_mhz = channel
                    point_rsrps.append(antenna.rsrp(point, 1))
                channel_rsrps.append(point_rsrps)
            antenna_rsrps.append(channel_rsrps)
        return antenna_rsrps

base_stations = []
channels = [3600, 3700]
points = gen_random_points(500)
for _ in range(6):
    base_stations.append(BaseStation(2, channels))

print('BS Index, Antenna Index, Channel Index, Lat, Lon, RSRP')
for bs_index, bs in enumerate(base_stations):
    bs_rsrps = bs.get_rsrps(points)
    for ant_index, ant_rsrps in enumerate(bs_rsrps):
        for chan_index, chan_rsrps in enumerate(ant_rsrps):
            for point_index, rsrp in enumerate(chan_rsrps):
                lat = points[point_index].y
                lon = points[point_index].x
                print(bs_index, ant_index, chan_index, lat, lon, rsrp, sep=', ')

print('BS Index, Antenna Index, Lat, Lon, Azimuth')
for bs_index, bs in enumerate(base_stations):
    for ant_index, antenna in enumerate(bs.antennas):
        lat = antenna.pos.y
        lon = antenna.pos.x
        azimuth = antenna.azimuth
        print(bs_index, ant_index, lat, lon, azimuth, sep=', ')
