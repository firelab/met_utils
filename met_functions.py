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
"""Angular distance traveled by the earth around the sun in one day, (365 days/yr)"""
HALF_PI   = math.pi/2.0 * u.rad
MINDECL = -0.4092797 * u.rad
DAYSOFF = 11.25 * u.day
INV_EARTH_OMEGA = (1*u.day)/(360*u.deg)
"""Inverse of Earth's angular velocity around its own axis"""
#INV_EARTH_OMEGA = 13750.9871 * u.s / u.rad

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
    
    MT-CLIM version 4.3 (http://www.ntsg.umt.edu/project/mtclim) does not
    provide any indication as to the source of the algorithm. It specifies 
    that the daylength calculation was "improved" between version 2.0 and 3.1,
    and indicates only that daylength is "sunrise to sunset, flat horizons".
    Neither sunrise nor sunset is defined.
    
    This python version agrees numerically with the C version to the 
    precision of the provided test case (0.5 seconds). It does not agree with
    US Naval Observatory daylength tables to a precision of 15 minutes, but 
    does agree to a precision of 20 minutes. The US Naval Observatory tables
    consider daylength to be the length of time that any portion of the solar
    disk is above the horizon.
    
    To keep things sane, we expect either lat or yday may be an array, but the
    other should be a scalar. The returned array should correspond to whichever
    input variable was an array. Providing arrays for both lat and yday is 
    very likely to fail, and on the off chance that you provided same-length
    arrays, the lat, yday, and result arrays should be treated as parallel data
    items (the first element in lat and yday produced the first element in the 
    returned array.)"""

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
    return 2.0 * hss * INV_EARTH_OMEGA
    
## \fn CalcDaylightHours(self,Julian)
## \brief Calculate Daylight hours as a function of Latitude and Julian Date
## \param Julian Julian Year Day (1-366)
## Requires that Latitude already be set as part of the class intialization
def CalcDaylightHours(Lat,Julian):
    """Returns hours of daylight given latitude and day of year
    
    This code is from Matt Jolly's South Africa project, adapted to use
    numpy, astropy.units, and the constants in this file. Previous incarnations
    existed in both C++ and python form. The source of the algorithm itself is uncertain.
    
    Similarity to the above indicates that both processes are engaged in 
    calculating the hour angle at sunset, then multiplying by twice the 
    INV_EARTH_OMEGA constant, which is the inverse of earth's angular velocity 
    around its own axis.
    
    Differences in the empirical constants and the algorithm lead to maximum
    differences of 20 minutes between the two methods for 48 deg N. In the following,
    the first column is the day of year, second is "CalcDaylightHours-calc_dayl"
    
    1.0 d : 17.8229292017 min
    11.0 d : 14.2908873325 min
    21.0 d : 10.0640795076 min
    31.0 d : 5.96029096671 min
    41.0 d : 2.64652323861 min
    51.0 d : 0.495394703635 min
    61.0 d : -0.451881890049 min
    71.0 d : -0.444762794206 min
    81.0 d : -8.16628769075 min
    91.0 d : 0.510771647146 min
    101.0 d : 0.358099842218 min
    111.0 d : -0.818598921154 min
    121.0 d : -3.21648709619 min
    131.0 d : -6.72941802256 min
    141.0 d : -10.9198618525 min
    151.0 d : -15.0755598443 min
    161.0 d : -18.3752109107 min
    171.0 d : -20.1411225478 min
    181.0 d : -20.0845166068 min
    191.0 d : -18.4137886715 min
    201.0 d : -15.7400061159 min
    211.0 d : -12.8384405337 min
    221.0 d : -10.3946988067 min
    231.0 d : -8.83346374486 min
    241.0 d : -8.25596439106 min
    251.0 d : -8.46385132721 min
    261.0 d : -9.97505928708 min
    271.0 d : -9.42572758511 min
    281.0 d : -9.0797579971 min
    291.0 d : -7.54290641036 min
    301.0 d : -4.5676427783 min
    311.0 d : -0.200220036887 min
    321.0 d : 5.17484587196 min
    331.0 d : 10.8662852045 min
    341.0 d : 16.0138912527 min
    351.0 d : 19.8265890555 min

    """
    phi = np.tan(Lat) * -1.0
    
    # following line treats one day as if it were exactly one degree.
    # this is in error by 5 parts in 360 and causes discontinuities. 
    # should be multiplied by RADPERDAY instead of unity.
    xfact = (Julian - (80*u.day)) * (u.deg/u.day)
    
    decl = (23.5*u.deg) * np.sin(xfact)
    tla = phi * np.sin(decl)

    np.clip(tla, -0.999999, 0.999999)
    
    try : 
        tla[np.abs(tla) < 0.01] = 0.01
    except: 
        if (np.abs(tla) < .01):
            tla = 0.01			
		
    tla = np.arctan(np.sqrt((1.0 - tla * tla))/tla)
    
    try :
        tla[tla<0.0] += (3.141593 * u.rad)
    except : 
        if (tla < 0.0):
            tla = tla + 3.141593
            
    return tla * 2 * INV_EARTH_OMEGA

    
def calc_vp_spec_humidity(q, p) :
    """ORCHIDEE inputs have specific humidity "q" (g/g) and pressure "p" (Pa). 
    Technically, the units of pressure don't matter, as the return value will
    be expressed in whatever pressure units you use.
    
    This is equation 2.19 (p. 17) of [1], solved for vapor pressure.
    
    Re
    
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
    
    Returns a relative humidity as a fraction 0..1, not as a percent 0..100.
    
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

           