import os
import math
import re
import requests
import warnings
import geopandas as gpd
import laspy
import rasterio
import rasterio.warp
import rasterio.transform
import numpy as np

from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from pyproj import Transformer

# This warning can be ignored since the insecure requests are only being made to a public API
warnings.simplefilter(
    'ignore',
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)

def get_poly_str(gdf):
    """
    Get textual description of the polygon bounds to be passed to the API
    """
    bounding_poly = gdf.iloc[0]['geometry']
    poly_str = ''
    for point in bounding_poly.exterior.coords:
        point_str = ',' + str(point[0]) + ' ' + str(point[1])
        poly_str = poly_str + point_str
    return poly_str[1:]

def download_file(url, output_dir):
    """
    Download a file from the specified URL
    """
    output_path = os.path.join(output_dir, re.search(r'([^/]+)$', url).group(1))
    response = requests.get(url, verify=False)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, 'wb') as file:
        file.write(response.content)

def download_files(urls, output_dir):
    """
    Download the file from each URL in the list
    """
    for i, url in enumerate(urls):
        print(f'\tDownloading file {i + 1} of {len(urls)}...')
        download_file(url, output_dir)
    if len(urls) == 0:
        print('\tAll required scans are already present.')

class LidarPointCloud:
    def __init__(self, path):
        las = laspy.read(path)

        # Create a transformer to convert coordinates to their projection
        crs = las.header.parse_crs().to_wkt()
        self.transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)

        # Get the bounding polygon
        minx, maxx = las.header.min[0], las.header.max[0]
        miny, maxy = las.header.min[1], las.header.max[1]
        self.bounding_poly = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])

        # Store X and Y coordinates in a KD-tree for quick indexing
        x_coords = las.x
        y_coords = las.y
        self.xy_tree = cKDTree(np.column_stack((x_coords, y_coords)))
        self.elevations = las.z

    def get_elevation(self, x, y):
        """
        Find the elevation at the defined point closest to the one provided
        """
        index = self.xy_tree.query([x, y])[1]
        return self.elevations[index]

def get_lidar_urls(gdf):
    """
    Get the download link to each LiDAR scan in the coverage area
    """
    poly_str = get_poly_str(gdf)
    url = 'https://tnmaccess.nationalmap.gov/api/v1/products'
    params = {'polygon': poly_str, 'datasets': 'Lidar Point Cloud (LPC)'}
    response = requests.get(url, params=params).json()
    urls = []
    for item in response['items']:
        urls.append(item['downloadURL'])
    return urls


def parse_lpcs(laz_paths):
    """
    Load each .laz file and store the contents in a LidarPointCloud object
    """
    print('--- Loading LiDAR scans into memory...')
    lpcs = []
    for index, path in enumerate(laz_paths):
        print(f'\tLoading file {index + 1} of {len(laz_paths)}...')
        lpcs.append(LidarPointCloud(path))
    return lpcs

def load_all_lpcs(gdf, output_dir):
    """
    Download all bounded .laz files and parse them as LidarPointCloud objects
    """
    print('--- Downloading LiDAR scans from the USGS database...')
    urls = get_lidar_urls(gdf)
    missing_urls = []
    laz_paths = []
    for url in urls:
        path = os.path.join(output_dir, re.search(r'([^/]+)$', url).group(1))
        if not os.path.exists(path):
            missing_urls.append(url)
        laz_paths.append(path)
    download_files(missing_urls, output_dir)
    return parse_lpcs(laz_paths)

def get_lpc_elevation(lpcs, pos, proj=True):
    """
    Get the elevation at the latitude/longitude pair from the appropriate LPC
    """
    x, y = pos.x, pos.y
    if not proj:
        transformer = lpcs[0].transformer
        x_pos, y_pos = transformer.transform(x, y)
    else:
        x_pos, y_pos = x, y
    for lpc in lpcs:
        if lpc.bounding_poly.buffer(1).intersects(Point(x_pos, y_pos)):
            return lpc.get_elevation(x_pos, y_pos)

def make_profile(lpcs, tx_pos, rx_pos, granularity):
    """
    Generate a surface profile from Tx to Rx with LPC data
    """
    transformer = lpcs[0].transformer  # Assuming all LPCs are using the same projection
    tx_lon, tx_lat = tx_pos.x, tx_pos.y
    rx_lon, rx_lat = rx_pos.x, rx_pos.y
    tx_x, tx_y = transformer.transform(tx_lon, tx_lat)
    rx_x, rx_y = transformer.transform(rx_lon, rx_lat)
    dx = abs(tx_x - rx_x)
    dy = abs(tx_y - rx_y)
    rx_distance = math.sqrt(dx ** 2 + dy ** 2) / 1000
    num_points = int(rx_distance // (granularity / 1000))
    x_coords = np.linspace(tx_x, rx_x, num=num_points)
    y_coords = np.linspace(tx_y, rx_y, num=num_points)
    distances = np.linspace(0, rx_distance, num=num_points)
    profile = list(map(
        lambda point: [point[2], get_lpc_elevation(lpcs, Point(point[0], point[1]))],
        list(zip(x_coords[1:], y_coords[1:], distances[1:]))
    ))
    return profile

class TopoMap:
    def __init__(self, path):
        self.topo = rasterio.open(path)
        self.transformer = rasterio.transform.AffineTransformer(self.topo.transform)
        bounds = self.topo.bounds
        minx, maxx, miny, maxy = rasterio.warp.transform_bounds(self.topo.crs.to_epsg(), 4326, *bounds)
        self.bounding_poly = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])

    def get_elevation(self, point):
        """
        Find the elevation at the defined point closest to the one provided
        """
        row, col = self.transformer.rowcol(point.x, point.y)
        return self.topo.read(1)[row, col]

def get_topo_urls(gdf):
    """
    Get the download link to each 10m topographic map in the coverage area
    """
    poly_str = get_poly_str(gdf)
    url = 'https://tnmaccess.nationalmap.gov/api/v1/products'
    params = {'polygon': poly_str, 'datasets': 'National Elevation Dataset (NED) 1/3 arc-second'}
    response = requests.get(url, params=params).json()
    urls = []
    for item in response['items']:
        urls.append(item['downloadURL'])
    return urls

def load_all_topos(gdf, output_dir):
    """
    Ensure that all necessary topographical maps are present in the expected directory
    """
    urls = get_topo_urls(gdf)
    missing_urls = []
    topo_paths = []
    for url in urls:
        path = os.path.join(output_dir, re.search(r'([^/]+)$', url).group(1))
        if not os.path.exists(path):
            missing_urls.append(url)
        topo_paths.append(path)
    download_files(missing_urls, output_dir)
    return list(map(lambda path: TopoMap(path), topo_paths))

def get_base_elevation(topos, point):
    """
    Get the base elevation (not including building height), at the given point
    """
    for topo in topos:
        if topo.bounding_poly.buffer(1).intersects(point):
            return topo.get_elevation(point)