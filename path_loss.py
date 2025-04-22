import math
import numpy as np
import geopandas as gpd

from shapely.geometry import Point
from shapely import wkt
from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
from itmlogic.misc.qerfi import qerfi
from itmlogic.statistics.avar import avar
from surface_profile import get_elevation, make_surface_profile

def free_space_pathloss(params):
    distance = params['distance']
    f_mhz    = params['f_mhz']
    return 20 * math.log10((4 * math.pi * distance * f_mhz * 1E6) / 3E8)

def alpha_beta_pathloss(params):
    distance = params['distance']
    alpha    = params['alpha']
    beta     = params['beta']
    return alpha + 10 * beta * math.log10(distance)

def ehata_pathloss(params):
    distance    = params['distance']
    f_mhz       = params['f_mhz']
    ant_height  = params['ant_height']
    ue_height   = params['ue_height']
    environment = params['environment']

    a = 69.55 + 26.16 * math.log10(f_mhz) - 13.82 * math.log10(ant_height)
    b = (44.9 - 6.55 * math.log10(ue_height)) * math.log10(distance / 1000)
    c = 3.2 * (math.log10(11.75 * ue_height)) ** 2 - 4.97
    if environment == 'urban':
        return a + b - c
    else:
        correction = 2 * (math.log10(f_mhz / 28)) ** 2 + 5.4
        return a + b - c - correction

def log_distance_pathloss(params):
    distance  = params['distance']
    f_mhz     = params['f_mhz']
    exponent  = params['exponent']
    shadowing = params['shadowing']
    omni      = params['omni']

    ld = 20 * math.log10((4 * math.pi * f_mhz * 1E6) / 3E8) + 10 * exponent * math.log10(distance)
    if shadowing:
        sigma = 13.15 if omni else 14.74
        return ld + sigma * np.random.randn()
    else:
        return ld

def itm_pathloss(params):
    distance   = params['distance']
    f_mhz      = params['f_mhz']
    rx_pos     = params['rx_pos']
    rx_height  = params['rx_height']
    tx_pos     = params['tx_pos']
    rx_pos     = params['rx_pos']
    tx_height  = params['tx_height']
    lpcs       = params['lpcs']
    confidence = params['confidence']

    # List antenna heights, accounting for base elevation
    tx_height_adjusted = get_elevation(lpcs, tx_pos.x, tx_pos.y, proj=False) + tx_height
    rx_height_adjusted = get_elevation(lpcs, rx_pos.x, rx_pos.y, proj=False) + rx_height
    antenna_heights = [tx_height_adjusted, rx_height_adjusted]

    prop = {}
    prop['f_mhz'] = f_mhz             # Frequency of the signal in MHz
    prop['d'] = distance / 1000       # Distance between points in km
    prop['hg'] = antenna_heights      # Tx and Rx heights
    prop['eps'] = 5                   # Terrain relative permittivity
    prop['sgm'] = 0.05                # Terrain conductivity (S/m)
    prop['klim'] = 5                  # Climate selection
    prop['ens0']  = 314               # Surface refractivity
    prop['lvar'] = 5                  # Initial value for AVAR control paremeter
    prop['gma']  = 157E-9             # Inverse Earth radius
    prop['wn'] = prop['f_mhz'] / 47.7 # Initialize wave number
    prop['ens'] = prop['ens0']        # Initialize refractive index properties
    prop['ipol'] = 0                  # Polarization code
    prop['kwx'] = 0                   # Set error marker to 0 initially

    # Include refraction in the effective Earth curvature parameter
    prop['gme'] = prop['gma'] * (1 - 0.04665 * math.exp(prop['ens'] / 179.3))

    # Surface transfer impedence
    prop['zgnd'] = np.sqrt(complex(prop['eps'], 376.62 * prop['sgm'] / prop['wn']) - 1)

    # Used internally for the avar routine initialization
    prop['klimx'] = 0
    prop['mdvarx'] = 11

    # Generate a surface profile
    if params['surface_profile'] == None:
        params['surface_profile'] = make_surface_profile(
            lpcs,                # List of LiDAR point clouds
            tx_pos.y, tx_pos.x,  # Tx latitude/longitude
            rx_pos.y, rx_pos.x,  # Rx latitude/longitude
            1                    # 1 meter granularity
        )
        surface_profile = params['surface_profile']
    else:
        surface_profile = params['surface_profile']

    # Set up the path profile to match the required format
    pfl = [len(surface_profile) - 1]
    pfl.append(prop['d'] * 1000 / pfl[0])
    for elevation in surface_profile:
        pfl.append(elevation)

    # Store path profile
    prop['pfl'] = pfl 

    qc = [confidence]  # Confidence  levels for predictions
    qr = [50]          # Reliability levels for predictions

    # Convert requested qc and qr into arguments of standard normal distribution
    zr = qerfi([x / 100 for x in qr])
    zc = qerfi([x / 100 for x in qc])

    # Initialize point-to-point mode parameters
    prop = qlrpfl(prop)

    # Calculate free space path loss
    fs = 8.685890 * np.log(2 * prop['wn'] * prop['dist'])

    # With confidence `qc', attenuation will not exceed `a' with a reliability >= `qr' 
    a, prop = avar(zr[0], 0, zc[0], prop)

    # Path loss is the sum of free space path loss, and the correction
    return a + fs

class PathlossModel:
    def __init__(self, pathloss_func, params):
        self.pathloss_func = pathloss_func
        self.params = params
    
    def pathloss(self, f_mhz, distance, tx_pos, tx_height, rx_pos, rx_height,
                 reuse_pfl=False, update_confidence=None):
        self.params['f_mhz']     = f_mhz
        self.params['distance']  = math.sqrt(distance ** 2 + (tx_height - rx_height) ** 2)
        self.params['tx_pos']    = tx_pos
        self.params['rx_pos']    = rx_pos
        self.params['tx_height'] = tx_height
        self.params['rx_height'] = rx_height
        if not(reuse_pfl):
            self.params['surface_profile'] = None
        if update_confidence != None:
            self.params['confidence'] = update_confidence
        return self.pathloss_func(self.params)

class FreeSpaceModel(PathlossModel):
    def __init__(self):
        super().__init__(free_space_pathloss, {})

class AlphaBetaModel(PathlossModel):
    def __init__(self, alpha, beta):
        params = {'alpha': alpha, 'beta': beta}
        super().__init__(alpha_beta_pathloss, params)

class ExtendedHataModel(PathlossModel):
    def __init__(self, environment):
        params = {'environment': environment}
        super().__init__(ehata_pathloss, params)

class LogDistanceModel(PathlossModel):
    def __init__(self, exponent, shadowing, omni):
        params = {'exponent': exponent, 'shadowing': shadowing, 'omni': omni}
        super().__init__(log_distance_pathloss, params)

class ItmModel(PathlossModel):
    def __init__(self, lpcs, confidence):
        params = {'lpcs': lpcs, 'confidence': confidence}
        super().__init__(itm_pathloss, params)
