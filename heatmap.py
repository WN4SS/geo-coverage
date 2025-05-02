import antenna
import math
import multiprocessing
import itertools
import pickle
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import contextily as ctx
import surface_profile
import path_loss

from shapely.geometry import Point
from antenna import Antenna
from pyproj import Transformer

def get_coverage_area(gdf):
    return gdf.iloc[0]['geometry']

def get_antennas(gdf, pattern_h, pattern_v):
    antennas = []
    for i in range(1, len(gdf)):
        antenna = Antenna(
            gdf.iloc[i]['frequency'],
            gdf.iloc[i]['Ptx'],
            gdf.iloc[i]['geometry'],
            gdf.iloc[i]['height'],
            gdf.iloc[i]['azimuth'], 1,
            pattern_h, pattern_v
        )
        antennas.append(antenna)
    return antennas

def get_projection(point):
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    x, y = transformer.transform(point.x, point.y)
    return Point(x, y)

def get_lat_lon(point):
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
    return list(map(
        lambda point: surface_profile.make_profile(lpcs, pos, point, granularity),
        rx_points
    ))

def get_profiles_multicore(num_cores, lpcs, rx_points, pos, granularity):
    rx_points_chunked = np.array_split(rx_points, num_cores)
    arglist = map(lambda chunk: [lpcs, chunk, pos, granularity], rx_points_chunked)
    with multiprocessing.Pool(processes=num_cores) as pool:
        segments = pool.starmap(get_profiles, arglist)
    return list(itertools.chain.from_iterable(segments))

def get_coverage_map(antenna, rx_points, profiles):
    rsrps = []
    for point, profile in zip(rx_points, profiles):
        rsrp = path_loss.rsrp(antenna, point, profile)
        rsrps.append([point, rsrp])
    return rsrps

def combined_coverage_map(antennas, rx_points, profiles):
    maps = []
    for antenna in antennas:
        maps.append(get_coverage_map(antenna, rx_points, profiles))
    best_rsrps = []
    for index in range(len(maps[0]) - 1):
        best_rsrp = [None, float('-inf')]
        for map in maps:
            if map[index][1] > best_rsrp[1]:
                best_rsrp = map[index]
        best_rsrps.append(best_rsrp)
    return best_rsrps

def plot_antenna(antenna, ax):
    proj = get_projection(antenna.pos)
    angle = math.radians(90 - antenna.azimuth)
    start_x = proj.x
    start_y = proj.y
    end_x = start_x + 500 * math.cos(angle)
    end_y = start_y + 500 * math.sin(angle)
    ax.annotate('', xytext=(start_x, start_y), xy=(end_x, end_y),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))

def plot_coverage_map(coverage_map, coverage_area, antennas, title):
    # Insert basemap into the plot
    sw_bound, ne_bound = get_coverage_bounds(coverage_area)
    sw_proj = get_projection(sw_bound)
    ne_proj = get_projection(ne_bound)
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(sw_proj.x - 100, ne_proj.x + 500)
    ax.set_ylim(sw_proj.y - 100, ne_proj.y + 500)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Plot coverage area bounding box
    area_gdf = gpd.GeoDataFrame(geometry=[coverage_area], crs='EPSG:4326')
    area_gdf = area_gdf.to_crs('EPSG:3857')
    area_gdf.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='none')
    
    # Plot sampled points
    colormap = matplotlib.cm.RdYlGn
    norm = matplotlib.colors.Normalize(vmin=-120, vmax=-80)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(sm, ax=ax, label='RSRP (dBm)')
    points = list(map(lambda point: gpd.GeoDataFrame(geometry=[point[0]], crs='EPSG:4326').to_crs('EPSG:3857').geometry,
                      coverage_map))
    colors = list(map(lambda point: colormap(norm(point[1])), coverage_map))
    x = [point.x for point in points]
    y = [point.y for point in points]
    print('plotting')
    print(len(x), len(y))
    ax.scatter(x, y, rasterized=True, c=colors, marker='s', s=40, alpha=0.5)

    # Plot antennas
    for antenna in antennas:
        plot_antenna(antenna, ax)

    ax.set_axis_off()
    plt.title(title)
    plt.savefig('test.png')

print('--- Opening scenario...')
gdf = gpd.read_file('scenarios/fruit-belt.geojson')

print('--- Loading antenna pattern...')
pattern_h = antenna.load_pattern('patterns/horizontal.pat')
pattern_v = antenna.load_pattern('patterns/vertical.pat')

print('--- Configuring antennas...')
antennas = get_antennas(gdf, pattern_h, pattern_v)

print('--- Generating coverage points...')
coverage_area = get_coverage_area(gdf)
rx_points = get_rx_points(coverage_area, 9)

print('--- Loading surface profiles...')
with open('profiles/9m-2m-fruit-belt.pkl', 'rb') as file:
    profiles = pickle.load(file)

"""
lpcs = surface_profile.load_all_lpcs(gdf, 'lidar')
print('--- Computing surface profiles...')
profiles = get_profiles_multicore(64, lpcs, rx_points, antennas[0].pos, 2)

print('--- Storing computed surface profiles...')
with open('9m-2m-fruit-belt.pkl', 'wb') as file:
    pickle.dump(profiles, file)
"""

print('--- Calculating estimated coverage map...')
coverage_map = combined_coverage_map(antennas, rx_points, profiles)

print('--- Plotting estimated coverage map...')
plot_coverage_map(coverage_map, coverage_area, antennas, 'Canandaigua: 9m granularity, 2m profile granularity')
