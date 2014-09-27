# -*- coding: utf-8 -*-
"""Implements growing season index calculations.

For more information see [1]

[1] Jolly, William M., Ramakrishna Nemani, and Steven W. Running. 2005. 
    “A Generalized, Bioclimatic Index to Predict Foliar Phenology in Response 
    to Climate.” Global Change Biology 11 (4): 619–32. 
    doi:10.1111/j.1365-2486.2005.00930.x. 
"""

import numpy as np
from astropy import units as u

# these constants affect the scaling of various met variables.
# Values are taken from [1].
#
# [1] Jolly, William M., Ramakrishna Nemani, and Steven W. Running. 2005. 
#     “A Generalized, Bioclimatic Index to Predict Foliar Phenology in 
#     Response to Climate.” Global Change Biology 11 (4): 619–32. 
#     doi:10.1111/j.1365-2486.2005.00930.x.
TMIN_MIN = -2. * u.deg_C
TMIN_MAX =  5. * u.deg_C
VPD_MIN  = 900. * u.Pa
VPD_MAX  = 4100. * u.Pa
PHOTO_MIN = 10. * u.hour
PHOTO_MAX = 11. * u.hour


class Normalize (object) : 
    """Class scales values from 0 to 1 given a min and a max"""
    def __init__(self, minval, maxval) : 
        self._minval = minval
        self._maxval = maxval
        
    def scale(self, x) : 
        scaled_x = (x - self._minval) / (self._maxval - self._minval)
        
        # if x is an ndarry, do the clipping in place
        if (isinstance(x, np.ndarray)) : 
            scaled_x = np.clip(scaled_x, 0, 1, out=scaled_x)
        else : 
            scaled_x = np.clip(scaled_x, 0, 1)
            
        return scaled_x
                
def calc_gsi(tmin, vpd, photo) : 
    """calculate the growing season index
    
    input variables must be defined using the correct units or errors
    will result.
    
    Parameters
    ----------
    tmin : array : deg_C
        daily minimum temperature
    vpd  : array : Pa
        vapor pressure deficit
    photo : array : hours
        photoperiod: time from sunrise to sunset
        
    Returns
    -------
    gsi : array : dimensionless
        growing season index
    """
    tmin_i = __xf_tmin.scale(tmin)
    vpd_i  = 1 - __xf_vpd.scale(vpd)
    photo_i= __xf_photo.scale(photo)
    return tmin_i * vpd_i * photo_i

# Singleton, module private instances of the scaling relationships for use
# inside the GSI calculation        
__xf_tmin = Normalize(TMIN_MIN, TMIN_MAX)
__xf_vpd  = Normalize(VPD_MIN, VPD_MAX)
__xf_photo= Normalize(PHOTO_MIN, PHOTO_MAX)
