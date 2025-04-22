import math
import csv
import numpy as np
import geopandas as gpd
import path_loss

from shapely.geometry import Point
from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
from itmlogic.misc.qerfi import qerfi
from itmlogic.statistics.avar import avar

def calc_azimuth(tx_pos, rx_pos):
    lat_tx, lat_rx = math.radians(tx_pos.y), math.radians(rx_pos.y)
    delta_lon = math.radians(rx_pos.x - tx_pos.x)
    y = math.sin(delta_lon) * math.cos(lat_rx)
    x = (math.cos(lat_tx) * math.sin(lat_rx) -
         math.sin(lat_tx) * math.cos(lat_rx) * math.cos(delta_lon))
    azimuth_rad = math.atan2(y, x)
    return (math.degrees(azimuth_rad) + 360) % 360

def calc_downtilt(distance, tx_pos, tx_height, rx_pos, rx_height):
    delta_h = abs(tx_height - rx_height)
    return 90 - math.degrees(math.atan(delta_h / distance))

def load_pattern(path):
    with open(path, mode='r') as pattern_file:
        pattern = {}
        reader = csv.reader(pattern_file)
        for row in reader:
            pattern[float(row[0])] = float(row[1])
    return pattern;

class Antenna:
    def __init__(self, f_mhz, tx_power, pos, height, azimuth, downtilt, pattern_h, pattern_v,
                 pathloss_model):
        self.f_mhz = f_mhz                    # Frequency at which the antenna transmits in MHz
        self.tx_power = tx_power              # Tx power in dBm
        self.pos = pos                        # Geographical coordinates at which the antenna is located
        self.height = height                  # Height at which the antenna is mounted
        self.azimuth = azimuth                # Azimuth at which the antenna is orientated in degrees
        self.downtilt = downtilt              # Downtilt of the antenna in degrees
        self.pattern_h = pattern_h            # Horizontal antenna gain pattern
        self.pattern_v = pattern_v            # Vertical antenna gain pattern
        self.pathloss_model = pathloss_model  # Model used for pathloss calculations

    def distance(self, rx_pos):
        lat_tx, lat_rx = math.radians(self.pos.y), math.radians(rx_pos.y)
        delta_lat = math.radians(self.pos.y - rx_pos.y)
        delta_lon = math.radians(self.pos.x - rx_pos.x)
        a = (math.sin(delta_lat / 2) ** 2 +
             math.sin(delta_lon / 2) ** 2 * math.cos(lat_tx) * math.cos(lat_rx))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        R = 6371000
        return R * c

    def power(self, rx_pos, rx_height):
        # Calculate the azimuth and downtilt between the antenna and the Rx
        azimuth = calc_azimuth(self.pos, rx_pos)
        downtilt = calc_downtilt(self.distance(rx_pos), self.pos, self.height, rx_pos, rx_height)

        # Translate the supplied azimuth to be in reference to the antenna azimuth
        azimuth_ref = azimuth - self.azimuth
        if azimuth_ref < 0:
            azimuth_ref = azimuth_ref + 360

        # Translate the supplied downtilt to be in reference to the antenna downtilt
        downtilt_ref = downtilt + self.downtilt

        # Get a list of all azimuths and downtilts for which gain has been defined
        defined_azimuths = list(self.pattern_h.keys())
        defined_downtilts = list(self.pattern_v.keys())

        # Find the azimuth and downtilt values that are closest to those requested
        closest_azimuth = min(defined_azimuths, key=lambda x: abs(x - azimuth_ref))
        closest_downtilt = min(defined_downtilts, key=lambda x: abs(x - downtilt_ref))

        # Get the linear horizontal and vertical gains based
        linear_gain_h = 10 ** (self.pattern_h[closest_azimuth] / 10)
        linear_gain_v = 10 ** (self.pattern_v[closest_downtilt] / 10)

        # Compute the total linear gain and convert to logarithmic
        total_linear_gain = math.sqrt(linear_gain_h ** 2 + linear_gain_v **2)
        total_gain = 10 * math.log10(total_linear_gain)
        return self.tx_power + total_gain
    
    def pathloss(self, rx_pos, rx_height, reuse_pfl=False, update_confidence=None):
        return self.pathloss_model.pathloss(self.f_mhz, self.distance(rx_pos), self.pos, self.height,
                                            rx_pos, rx_height, reuse_pfl, update_confidence)

    def rsrp(self, rx_pos, rx_height, reuse_pfl=False, update_confidence=None):
        pathloss = self.pathloss(rx_pos, rx_height, reuse_pfl, update_confidence)
        return self.power(rx_pos, rx_height) - pathloss
    
    def rx_prob(self, rx_pos, rx_height):
        floor = -95 + 10
        prob = 70 
        if self.rsrp(rx_pos, rx_height, False, 70) <= floor:
            return prob
        confidences = np.linspace(99, 71, 99)
        for confidence in confidences:
            rsrp = self.rsrp(rx_pos, rx_height, True, confidence)
            if rsrp >= floor:
                prob = confidence
                break
        return prob
