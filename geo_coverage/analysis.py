from .antenna import Antenna, load_pattern
from .pathloss import rsrp
from .elevation import load_all_lpcs, make_profile
from .elevation import load_all_topos, get_base_elevation

from shapely.geometry import Point
from pyproj import Transformer

import numpy as np
import geopandas as gpd
import itertools
import pickle
import multiprocessing
import os

def get_antennas(gdf, pattern_h, pattern_v, topos):
    """
    Load the antennas defined in the supplied GeoJSON into a list of antenna objects
    """
    antennas = []
    for i in range(1, len(gdf)):
        pos = gdf.iloc[i]['geometry']
        base_elevation = get_base_elevation(topos, pos)
        antenna = Antenna(
            gdf.iloc[i]['frequency'],
            gdf.iloc[i]['Ptx'],
            pos,
            gdf.iloc[i]['height'] + base_elevation,
            gdf.iloc[i]['azimuth'], 5,
            pattern_h, pattern_v
        )
        antennas.append(antenna)
    return antennas

def get_projection(point):
    """
    Convert the supplied point from latitude/longitude to Web Mercator
    """
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    x, y = transformer.transform(point.x, point.y)
    return Point(x, y)

def get_lat_lon(point):
    """
    Convert the suppplied point from Web Mercator to latitude/longitude
    """
    transformer = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
    x, y = transformer.transform(point.x, point.y)
    return Point(x, y)

def get_coverage_bounds(coverage_area):
    sw_bound_lat, sw_bound_lon = [float('inf')] * 2
    ne_bound_lat, ne_bound_lon = [float('-inf')] * 2
    for lon, lat in coverage_area.exterior.coords:
        if lat < sw_bound_lat:
            sw_bound_lat = lat
        if lat > ne_bound_lat:
            ne_bound_lat = lat
        if lon < sw_bound_lon:
            sw_bound_lon = lon
        if lon > ne_bound_lon:
            ne_bound_lon = lon
    return [Point(sw_bound_lon, sw_bound_lat), Point(ne_bound_lon, ne_bound_lat)]

def get_rx_points(coverage_area, granularity):
    """
    Generate a grid of coverage points within the supplied area
    """
    sw_bound, ne_bound = get_coverage_bounds(coverage_area)
    sw_proj = get_projection(sw_bound)
    ne_proj = get_projection(ne_bound)
    x_coords = np.arange(sw_proj.x, ne_proj.x, granularity)
    y_coords = np.arange(sw_proj.y, ne_proj.y, granularity)
    rx_points = []
    for x in x_coords:
        for y in y_coords:
            point = get_lat_lon(Point(x, y))
            if coverage_area.contains(point):
                rx_points.append(point)
    return rx_points

def get_profiles(lpcs, rx_points, pos, granularity):
    """
    Create a surface profile between a supplied position and every Rx point
    """
    return list(map(
        lambda point: make_profile(lpcs, pos, point, granularity),
        rx_points
    ))

def get_profiles_mc(num_cores, lpcs, rx_points, pos, granularity):
    """
    Use multiple cores to generate surface profiles
    """
    rx_points_chunked = np.array_split(rx_points, num_cores)
    arglist = map(lambda chunk: [lpcs, chunk, pos, granularity], rx_points_chunked)
    with multiprocessing.Pool(processes=num_cores) as pool:
        segments = pool.starmap(get_profiles, arglist)
    return list(itertools.chain.from_iterable(segments))

def get_base_elevations(topos, rx_points):
    """
    Get the base elevation excluding buildings at each Rx point
    """
    return list(map(lambda point: get_base_elevation(topos, point), rx_points))

def get_coverage_map(antenna, rx_points, profiles, base_elevations):
    """
    Calculate the estimated RSRP for each Rx point supplied
    """
    rsrps = []
    for point, profile, height in zip(rx_points, profiles, base_elevations):
        point_rsrp = rsrp(antenna, point, height, profile)
        rsrps.append([point, point_rsrp])
    return rsrps

def combined_coverage_map(antennas, rx_points, profiles, base_elevations):
    """
    Combine multiple coverage maps so that each point is associated with the best RSRP
    """
    maps = []
    for antenna in antennas:
        maps.append(get_coverage_map(antenna, rx_points, profiles, base_elevations))
    best_rsrps = []
    for index in range(len(maps[0]) - 1):
        best_rsrp = [None, float('-inf')]
        for map in maps:
            if map[index][1] > best_rsrp[1]:
                best_rsrp = map[index]
        best_rsrps.append(best_rsrp)
    return best_rsrps

def run_analysis(scenario, pattern, granularity, prof_granularity=2, num_cores=1):
    """
    Runs a coverage analysis with the given parameters, returning a coverage map
    """
    print('--- Initializing scenario...')
    gdf = gpd.read_file(os.path.join('scenarios', scenario + '.geojson'))
    print('--- Generating coverage points...')
    rx_points = get_rx_points(gdf.iloc[0]['geometry'], granularity)
    print('--- Loading antenna patterns...')
    pattern_h = load_pattern(os.path.join('patterns', pattern, 'horizontal.pat'))
    pattern_v = load_pattern(os.path.join('patterns', pattern, 'vertical.pat'))
    print('--- Downloading topographic maps from the USGS database...')
    topos = load_all_topos(gdf, 'topo')
    print('--- Loading base elevations...')
    base_elevations = get_base_elevations(topos, rx_points)
    antennas = get_antennas(gdf, pattern_h, pattern_v, topos)
    profiles_path = os.path.join('profiles', scenario + '.pkl')
    if not os.path.exists(profiles_path):
        if not os.path.exists('lidar'):
            os.makedirs('lidar')
        lpcs = load_all_lpcs(gdf, 'lidar')
        print('--- Computing surface profiles...')
        profiles = get_profiles_mc(num_cores, lpcs, rx_points, antennas[0].pos, prof_granularity)
        print('--- Storing precomputed profiles...')
        if not os.path.exists('profiles'):
            os.makedirs('profiles')
        with open(profiles_path, 'wb') as file:
            pickle.dump(profiles, file)
    else:
        print('--- Loading surface profiles...')
        with open(profiles_path, 'rb') as file:
            profiles = pickle.load(file)
    print('--- Calculating estimated coverage map...')
    return combined_coverage_map(antennas, rx_points, profiles, base_elevations), antennas, gdf
