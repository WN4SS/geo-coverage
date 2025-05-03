from .analysis import get_projection

from shapely.geometry import Point

import math
import matplotlib
import matplotlib.pyplot as plt
import contextily as ctx

def plot_antenna(antenna, ax):
    proj = get_projection(antenna.pos)
    angle = math.radians(90 - antenna.azimuth)
    start_x = proj.x
    start_y = proj.y
    end_x = start_x + 500 * math.cos(angle)
    end_y = start_y + 500 * math.sin(angle)
    ax.annotate('', xytext=(start_x, start_y), xy=(end_x, end_y),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))

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

def plot_coverage_map(coverage_map, coverage_area, antennas, title):
    """
    Plot a coverage map as a heatmap showing antenna positions overlayed an an OSM tile
    """
    sw_bound, ne_bound = get_coverage_bounds(coverage_area)
    sw_proj = get_projection(sw_bound)
    ne_proj = get_projection(ne_bound)
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(sw_proj.x - 100, ne_proj.x + 500)
    ax.set_ylim(sw_proj.y - 100, ne_proj.y + 500)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    area_gdf = gpd.GeoDataFrame(geometry=[coverage_area], crs='EPSG:4326')
    area_gdf = area_gdf.to_crs('EPSG:3857')
    area_gdf.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='none')
    
    colormap = matplotlib.cm.RdYlGn
    norm = matplotlib.colors.Normalize(vmin=-120, vmax=-80)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(sm, ax=ax, label='RSRP (dBm)')
    points = list(map(
        lambda pt: gpd.GeoDataFrame(geometry=[pt[0]], crs='EPSG:4326').to_crs('EPSG:3857').geometry,
        coverage_map)
    )
    colors = list(map(lambda point: colormap(norm(point[1])), coverage_map))
    x = [point.x for point in points]
    y = [point.y for point in points]
    ax.scatter(x, y, rasterized=True, c=colors, marker='s', s=40, alpha=0.5)
    for antenna in antennas:
        plot_antenna(antenna, ax)
    plt.title(title)
    plt.savefig(title + '.png')
