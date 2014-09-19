# -*- coding: utf-8 -*-
"""Module containing various meterological calculations, conversions,
and miscellaneous constants. Source for all equations is the met_functions.cpp
file which is part of FireFamilyPlus. Ultimate source is unknown.

The primary difference between this code and the original is that this code
is meant to use numpy array-like objects. Some methods may fail if passed scalars.
I have also added code specific to dealing with ORCHIDEE's typical meterological
inputs."""

import numpy as np
from  satvp import default as vp
import math

CELTOKEL = 273.15
"""Celcius to Kelvin conversion"""

# Constants for potential temperature and pressure calculation
LR_STD = 0.0065
T_STD = 288.15
R = 8.3143
MA = 28.9644e-3
P_STD = 101325.0
"""Standard value for sea level pressure (Pa)"""
G_STD = 9.80665
"""Standard value for gravitational acceleration (m/s^2)"""

# Radiation constants
RADPERDEG = 0.01745329
HALF_PI   = math.pi/2.0
MINDECL = -0.4092797
DAYSOFF = 11.25
SECPERRAD = 13750.9871

# constants for humidity
M_V = 18
"""Molar mass of water vapor (g/mol)"""
M_D = 28.964
"""Molar mass of dry air (g/mol), US Standard Atmosphere (1976) via CRC Handbook
of Chemistry and Physics, 77th edition 1996-1997, pp. 14-16"""
EPSILON = M_V/M_D
"""Ratio of water vapor to dry air molar mass."""





def calc_pres(elev) : 
    """Calculate pressure in Pascals given elevation in ???"""
    t1 = 1.0 - (LR_STD * elev) / T_STD
    t2 = G_STD / (LR_STD * ( R / MA ))
    pa = P_STD * pow(t1,t2)
    return pa

def calc_pot_temp(temp,elev) :
    """Calculates the potential temperature given ??? temperature and 
    elevation."""
    pa = calc_pres(elev)
    pa /= 100.0
    prat = 1000 / pa
    pottemp = (temp + CELTOKEL) * np.power(prat,0.286)
    return pottemp - CELTOKEL
    
def calc_inv_pot_temp(temp,elev):
    """???"""
    pa = calc_pres(elev)
    pa /= 100.0
    prat = 1000 / pa
    invpottemp = (temp + CELTOKEL) / (np.power(prat,0.286))
    return (invpottemp - CELTOKEL)



def calc_vpd(Tday, vp_act) :
    """Calculate vapor pressure deficit given the measured temperature and 
    actual (modeled) vapor pressure. Returns the VPD in Pa, limited to the 
    range [0-9000Pa]."""

    vp_sat = vp.calc_vp(Tday)
    vpd = vp_sat - vp_act
    if (vpd > 9000) :
        vpd = 9000.0
    if(vpd < 0) :
        vpd = 0.0
    return vpd

    

def calc_dayl(lat,yday):
    """Daylength function from MT-CLIM. Returns length of day in seconds
    given latitude in degrees and yday as integer day of year.
    
    To keep things sane, we expect lat may be an array, but yday should be 
    a scalar. We do not want to encourage the allocation of enough memory
    to hold daylengths for every pixel, every day of year."""

    # check for (+/-) 90 degrees latitude, throws off daylength calc 
    lat *= RADPERDEG
    np.clip(lat, -HALF_PI, HALF_PI, out=lat)
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    
    # calculate cos and sin of declination 
    decl = MINDECL * math.cos((yday + DAYSOFF) * RADPERDAY) # yday is scalar
    cosdecl = math.cos(decl)
    sindecl = math.sin(decl)
    
        
    # calculate daylength as a function of lat and decl 
    cosegeom = coslat * cosdecl
    sinegeom = sinlat * sindecl
    coshss = -(sinegeom) / cosegeom
    np.clip(coshss, -1.0, 1.0, out=coshss)
    hss = np.acos(coshss)              # hour angle at sunset (radians) 
    
    # daylength (seconds) 
    return 2.0 * hss * SECPERRAD;
    
def calc_vp_spec_humidity(q, p) :
    """ORCHIDEE inputs have specific humidity "q" (g/g) and pressure "p" (Pa). 
    Technically, the units of pressure don't matter, as the return value will
    be expressed in whatever pressure units you use.
    I got the raw material for this from lecture notes on the web. Good luck finding it: 
    http://www.geog.ucsb.edu/~joel/g266_s10/lecture_notes/chapt03/oh10_3_01/oh10_3_01.html
    """
    
    # calculate the mixing ratio
    w = q / (1.0 - q)
    
    return p * (w / (EPSILON + w))
    
    