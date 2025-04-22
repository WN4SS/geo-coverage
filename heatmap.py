import antenna
import math
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

def get_antennas(gdf, pathloss_model, pattern_h, pattern_v):
    antennas = []
    for i in range(1, len(gdf)):
        antenna = Antenna(gdf.iloc[i]['frequency'], gdf.iloc[i]['Ptx'], gdf.iloc[i]['geometry'],
                          gdf.iloc[i]['height'], gdf.iloc[i]['azimuth'], 1, pattern_h, pattern_v,
                          pathloss_model)
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

def get_coverage_map(antennas, rx_points):
    best_rsrps = [[Point(0, 0), float('-inf')]] * len(rx_points)
    for cur_ant, antenna in enumerate(antennas):
        rsrps = []
        for cur_pt, point in enumerate(rx_points):
            print(f'Antenna {cur_ant + 1}/{len(antennas)}: Point {cur_pt + 1} of {len(rx_points)}')
            rsrps.append([point, antenna.rsrp(point, 1)])
        for index in range(len(rx_points)):
            if(rsrps[index][1] > best_rsrps[index][1]):
                best_rsrps[index] = rsrps[index]
    return best_rsrps

def get_confidence_map(antennas, rx_points):
    highest_confidences = [[Point(0, 0), 0]] * len(rx_points)
    for cur_ant, antenna in enumerate(antennas):
        confidences = []
        for cur_pt, point in enumerate(rx_points):
            print(f'Antenna {cur_ant + 1}/{len(antennas)}: Point {cur_pt + 1} of {len(rx_points)}')
            confidences.append([point, antenna.rx_prob(point, 1)])
    for index in range(len(rx_points)):
        if(confidences[index][1] > highest_confidences[index][1]):
            highest_confidences[index] = confidences[index]
    return highest_confidences

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
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(sw_proj.x - 100, ne_proj.x + 500)
    ax.set_ylim(sw_proj.y - 100, ne_proj.y + 500)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Plot coverage area bounding box
    area_gdf = gpd.GeoDataFrame(geometry=[coverage_area], crs='EPSG:4326')
    area_gdf = area_gdf.to_crs('EPSG:3857')
    area_gdf.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='none')
    
    # Plot sampled points
    colormap = matplotlib.cm.RdYlGn
    norm = matplotlib.colors.Normalize(vmin=-100, vmax=-70)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(sm, ax=ax, label='RSRP (dBm)')
    for sample in coverage_map:
        color = colormap(norm(sample[1]))
        point_gdf = gpd.GeoDataFrame(geometry=[sample[0]], crs='EPSG:4326')
        point_gdf = point_gdf.to_crs('EPSG:3857')
        point_gdf.plot(ax=ax, color=color, marker='s', markersize=80, alpha=0.4)

    # Plot antennas
    for antenna in antennas:
        plot_antenna(antenna, ax)

    ax.set_axis_off()
    plt.title(title)
    plt.show()

def plot_confidence_map(confidence_map, coverage_area, antennas, title):
    # Insert basemap into the plot
    sw_bound, ne_bound = get_coverage_bounds(coverage_area)
    sw_proj = get_projection(sw_bound)
    ne_proj = get_projection(ne_bound)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(sw_proj.x, ne_proj.x)
    ax.set_ylim(sw_proj.y, ne_proj.y)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Plot coverage area bounding box
    area_gdf = gpd.GeoDataFrame(geometry=[coverage_area], crs='EPSG:4326')
    area_gdf = area_gdf.to_crs('EPSG:3857')
    area_gdf.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='none')

    # Plot sampled points
    colormap = matplotlib.cm.RdYlGn
    norm = matplotlib.colors.Normalize(vmin=70, vmax=99)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    cbar = plt.colorbar(sm, ax=ax, label='Signal Recoverable Confidence (%)')
    for sample in confidence_map:
        color = colormap(norm(sample[1]))
        point_gdf = gpd.GeoDataFrame(geometry=[sample[0]], crs='EPSG:4326')
        point_gdf = point_gdf.to_crs('EPSG:3857')
        point_gdf.plot(ax=ax, color=color, marker='s', markersize=110, alpha=0.4)

    # Plot antennas
    for antenna in antennas:
        plot_antenna(antenna, ax)

    ax.set_axis_off()
    plt.title(title)
    plt.show()

gdf = gpd.read_file('scenarios/canandaigua-small.geojson')
lpcs = surface_profile.load_all_lpcs(gdf, 'lidar')
model = path_loss.ItmModel(lpcs, 90)
pattern_h = antenna.load_pattern('patterns/horizontal.pat')
pattern_v = antenna.load_pattern('patterns/vertical.pat')
antennas = get_antennas(gdf, model, pattern_h, pattern_v)
coverage_area = get_coverage_area(gdf)
rx_points = get_rx_points(coverage_area, 50)

coverage_map = get_coverage_map(antennas, rx_points)
plot_coverage_map(coverage_map, coverage_area, antennas,
                  'Canandaigua Deployment: 90% Confidence, 50m Granularity, 1m Surface Profile Sampling Interval')
