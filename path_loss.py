import math

c_e = 1.57E-4 # Effective curvature of the earth in inverse km

def tx_visual_horizon(profile, h_ts, d_p):
    """
    Calculate the transmitter visual horizon
    """
    s_tm = None
    for sample in profile[1:-1]:
        d_i = sample[0]
        h_i = sample[1]
        s = (h_i - h_ts + 500 * c_e * d_i * (d_p - d_i)) / d_i
        if s_tm == None or s > s_tm:
            s_tm = s
    return s_tm

def rx_visual_horizon(profile, h_rs, d_p):
    """
    Calculate the receive visual horizon
    """
    s_rm = None
    for sample in profile[1:-1]:
        d_i = sample[0]
        h_i = sample[1]
        s = (h_i - h_rs + 500 * c_e * d_i * (d_p - d_i)) / (d_p - d_i)
        if s_rm == None or s > s_rm:
            s_rm = s
    return s_rm

def straight_line_path(h_ts, h_rs, d_p):
    """
    Calculate the straight line path from Tx to Rx
    """
    return (h_rs - h_ts) / d_p

def ked_param_los(profile, wavelength, h_ts, h_rs, d_p):
    """
    Calculate the KED approximation parameter for the LOS case
    """
    v_max = None
    for sample in profile[1:-1]:
        d_i = sample[0]
        h_i = sample[1]
        a = h_i + 500 * c_e * d_i * (d_p - d_i)
        b = (h_ts * (d_p - d_i) + h_rs * d_i) / d_p
        c = math.sqrt((0.002 * d_p) / (wavelength * d_i * (d_p - d_i)))
        v = (a - b) * c
        if v_max == None or v > v_max:
            v_max = v
    return v_max

def ked_param_transhorizon(wavelength, h_ts, h_rs, d_p, s_tm, s_rm):
    """
    Calculate the KED approximation parameter for the transhorizon case
    """
    d_b = (h_rs - h_ts + s_rm * d_p) / (s_tm + s_rm)
    a = h_ts + s_tm * d_b
    b = (h_ts * (d_p - d_b) + h_rs * d_b) / d_p
    c = math.sqrt((0.002 * d_p) / (wavelength * d_b * (d_p - d_b)))
    return (a - b) * c

def ked_approx(v):
    """
    Calculate the approximate KED diffraction loss in dB
    """
    return 6.9 + 20 * math.log10(math.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)

def corrected_loss(l_uc, d_p):
    """
    Calculate the corrected diffraction loss in dB
    """
    return l_uc + (1 - math.exp(-l_uc / 6)) * (10 + 0.02 * d_p)

def bullington_loss(profile, tx_height_rel, rx_height_rel, f_mhz):
    """
    Calculate the diffraction loss in dB with the Bullington model
    """
    if len(profile) < 3:
        return 0
    wavelength = 1 / (1E6 * f_mhz) 
    d_p = profile[len(profile) - 1][0]
    h_ts = profile[0][1] + tx_height_rel
    h_rs = profile[len(profile) - 1][1] + rx_height_rel
    s_tm = tx_visual_horizon(profile, h_ts, d_p)
    s_tr = straight_line_path(h_ts, h_rs, d_p)
    if s_tm <= s_tr:
        l_uc = ked_approx(ked_param_los(profile, wavelength, h_ts, h_rs, d_p))
    else:
        s_rm = rx_visual_horizon(profile, h_rs, d_p)
        l_uc = ked_approx(ked_param_transhorizon(wavelength, h_ts, h_rs, d_p, s_tm, s_rm))
    return corrected_loss(l_uc, d_p)

def free_space_loss(distance, f_mhz):
    """
    Calculate the free space path loss over the given distance
    """
    return 20 * math.log10(distance) + 20 * math.log10(f_mhz * 1E6) - 147.55 
   
def rsrp(antenna, rx_pos, profile):
    """
    Estimate path loss from an antenna given Rx position, and path profile
    """
    rx_height = profile[len(profile) - 1][1] + antenna.height_rel
    power = antenna.power(rx_pos, rx_height)
    lb = bullington_loss(profile, antenna.f_mhz)
    rx_distance = profile[len(profile) - 1][0] * 1000
    return power - max(lb, free_space_loss(rx_distance, antenna.f_mhz))
