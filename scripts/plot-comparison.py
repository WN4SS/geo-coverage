from shapely.geometry import Point

import matplotlib.pyplot as plt
import numpy as np

import sys
import xml.etree.ElementTree as et

import antenna
from antenna import Antenna
from path_loss import ItmModel

def feet_to_meters(feet):
	return feet / 3

# Load antenna pattern for 3.65Ghz 65 degree antenna
pattern_h = antenna.load_pattern('patterns/horizontal.pat')
pattern_v = antenna.load_pattern('patterns/vertical.pat')

# Position of the base station
bs_pos = Point(-77.279937, 42.888019)

pathloss_model = ItmModel('surfaces/canandaigua_10ft.npz', 95)

# South east antenna
police_se_height = feet_to_meters(100)
police_se_tx = Antenna(
	3560,                      # Frequency in MHz
	30,                        # Tx power in dBm
	bs_pos, police_se_height,  # Coordinates and height
	112, 1,                    # Azimuth and downtilt
	pattern_h, pattern_v,      # Antenna pattern
	pathloss_model)

# South west antenna
police_sw_height = feet_to_meters(95)
police_sw_tx = Antenna(
	3590,                      # Frequency in Mhz
	30,                        # Tx power in dBm
	bs_pos, police_sw_height,  # Coordinates and height
	182, 1,                    # Azimuth and downtilt
	pattern_h, pattern_v,      # Antenna pattern
	pathloss_model)

rx_pos = Point(-77.280033, 42.886169) # Chamber of Commerce
#rx_pos = Point(-77.277861, 42.884097) # Beeman Alley
#rx_pos = Point(-77.272381, 42.873011) # Pier
rx_height = 2 # Need more data on this

# Received signal power of each antenna at the receiver
police_se_rsrp = police_se_tx.rsrp(rx_pos, rx_height)
print('')
police_sw_rsrp = police_sw_tx.rsrp(rx_pos, rx_height)

# Print results
print('')
print(f'Police SE RSRP: {police_se_rsrp}')
print(f'Police SW RSRP: {police_sw_rsrp}')

# Parse the `Signal Hawk' spectrum scan XML
tree = et.parse('scans/chamber.xml')
root = tree.getroot()
params = root.find('parameterBean')
items = root.find('string-array')

# Get start and stop frequencies of the scan, and the number of data points
start_freq = float(params.find('startFrequency').text)
stop_freq = float(params.find('stopFrequency').text)
num_points = int(params.find('sweepPoint').text)

# Get a list of all frequencies sampled in the spectrum scan
freqs = np.linspace(start_freq, stop_freq, num_points).tolist()

# Get a list of all sampled power levels
power_levels = []
for item in items:
    power_levels.append(float(item.text))

# Replicate the spectrum scan plot
plt.plot(freqs, power_levels, label='Bird Data')
plt.title('Canandaigua Chamber of Commerce 3.0-3.7 Ghz')
plt.ylabel('Power Level: dBm')
plt.xlabel('Frequency')

# Plot RSRP from SE antenna
plt.plot([3.55e9, 3.57e9], [police_se_rsrp, police_se_rsrp], linewidth=2.5, color='r',
         label='Police SE antenna RSRP estimate')

# Plot RSRP from SW antenna
plt.plot([3.58e9, 3.60e9], [police_sw_rsrp, police_sw_rsrp], linewidth=2.5, color='g', 
         label='Police SW antenna RSRP estimate')

plt.legend(loc='upper left')

plt.show()
