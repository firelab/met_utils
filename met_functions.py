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
from astropy import units as u


# Radiation constants
RADPERDAY = 0.017214 * u.rad / u.day
HALF_PI   = math.pi/2.0 * u.rad
MINDECL = -0.4092797 * u.rad
DAYSOFF = 11.25 * u.day
SECPERRAD = 13750.9871 * u.s / u.rad

# constants for humidity
M_V = 18 * (u.g/u.mol)
"""Molar mass of water vapor (g/mol)"""
M_D = 28.964 * (u.g/u.mol)
"""Molar mass of dry air (g/mol), US Standard Atmosphere (1976) via CRC Handbook
of Chemistry and Physics, 77th edition 1996-1997, pp. 14-16"""
EPSILON = M_V/M_D
"""Ratio of water vapor to dry air molar mass."""





def calc_vpd(Tday, vp_act) :
    """Calculate vapor pressure deficit given the measured temperature and 
    actual (modeled) vapor pressure. Returns the VPD in Pa, limited to the 
    range [0-9000Pa]."""

    vp_sat = vp.calc_vp(Tday)
    vpd = vp_sat - vp_act
    np.clip(vpd, 0*u.Pa, 9000*u.Pa)
    return vpd

    

def calc_dayl(lat,yday):
    """Daylength function from MT-CLIM. Returns length of day in seconds
    given latitude in degrees and yday as integer day of year.
    
    To keep things sane, we expect lat may be an array, but yday should be 
    a scalar. We do not want to encourage the allocation of enough memory
    to hold daylengths for every pixel, every day of year."""

    # check for (+/-) 90 degrees latitude, throws off daylength calc 
    np.clip(lat, -HALF_PI, HALF_PI, out=lat)
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    
    # calculate cos and sin of declination 
    decl = MINDECL * np.cos((yday + DAYSOFF) * RADPERDAY) # yday is scalar
    cosdecl = np.cos(decl)
    sindecl = np.sin(decl)
    
        
    # calculate daylength as a function of lat and decl 
    cosegeom = coslat * cosdecl
    sinegeom = sinlat * sindecl
    coshss = -(sinegeom) / cosegeom
    np.clip(coshss, -1.0, 1.0, out=coshss)
    hss = np.arccos(coshss)              # hour angle at sunset (radians) 
    
    # daylength (seconds) 
    return 2.0 * hss * SECPERRAD
    
def calc_vp_spec_humidity(q, p) :
    """ORCHIDEE inputs have specific humidity "q" (g/g) and pressure "p" (Pa). 
    Technically, the units of pressure don't matter, as the return value will
    be expressed in whatever pressure units you use.
    
    This is equation 2.19 (p. 17) of [1], solved for vapor pressure.
    
    [1] Rogers, R. R. A Short Course in Cloud Physics, Third Edition. 
        3 edition. Oxford ; New York: Pergamon, 1989.
    """
    
    return (p*q)/(q + EPSILON - q*EPSILON)
    
def mixing_ratio(vp, p) :
    """Calculates the mixing ratio given the vapor pressure and the total pressure.
    
    Implements equation 2.18 (p.17) of [1].
    
    [1] Rogers, R. R. A Short Course in Cloud Physics, Third Edition. 
        3 edition. Oxford ; New York: Pergamon, 1989.
    """
    return EPSILON * (vp / (p - vp))
    

def calc_rh_spec_humidity(q, p, t) :
    """RH is needed for the various NFDRS fuel moistures.
    
    Implements equation 2.20 (p. 17) of [1].
    
    [1] Rogers, R. R. A Short Course in Cloud Physics, Third Edition. 
        3 edition. Oxford ; New York: Pergamon, 1989.    
    """
    e = calc_vp_spec_humidity(q,p)
    w = mixing_ratio(e,p)
    
    e_sat = vp.calc_vp(t)
    w_sat  = mixing_ratio(e_sat, p)
    
    return (w/w_sat).to(u.pct)
    

# Below here, functions were just copied out of the C code because they 
# were low hanging fruit. There is no current need for them, they have not
# been converted to use the units framework, and they are not tested.
# use at your own risk.

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

           