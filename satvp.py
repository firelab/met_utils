# -*- coding: utf-8 -*-
"""Classes and concrete calculators to express the relationships between 
saturation vapor pressue and dewpoint temperature. These are peer-reviewed.
If you don't care which one is best, just use "default", and use degrees C and 
Pascals."""

from abc import ABCMeta, abstractmethod
import numpy as np

class SaturationVapor (object) : 
    """Concrete subclasses implement a relationship between saturation vapor
    pressure and dewpoint temperature. A common feature to all calculation
    methods is that calc_vp(calc_tdew(temp)) == temp. In other words, the 
    two functions defined here are inverses."""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def calc_vp(self, temp) : 
        """Given the dewpoint temperature, calculate the saturation vapor 
        pressure"""
        pass
        
    @abstractmethod
    def calc_tdew(self, vp):
        """Given the saturation vapor pressure, calculate the dewpoint 
        temperature."""
        pass

class MagnusApproximation (SaturationVapor) : 
    """Abstract class defines the basic properties of the Magnus approximation.
    Contains the three coefficients and applicable 
    temperature range for the relationship."""
    
    __metaclass__ = ABCMeta
    
    def __init__(self, A, B, C, range) : 
        self._A = A
        self._B = B
        self._C = C
        self._range = range
        
    def get_range(self): 
        return self._range
    
class MagnusExp (MagnusApproximation) : 
    """Concrete class which implements a magnus approximation using natural
    logarithm and e."""
    
    def calc_vp(self, temp):
        """Implements: A * e^( B * temp / (C + temp))"""
        return self._A * np.exp((self._B * temp) / (self._C + temp))
        
    def calc_tdew(self, vp) :
        """Implements: C / ( (B/ln(vp/A)) - 1 )"""
        return self._C / (self._B/np.log(vp/self._A)-1)
        
class Magnus10 (MagnusApproximation) : 
    """Concrete class which implements a magnus approximation using log base 10
    and powers of 10"""
    
    def calc_vp(self, temp) :
        """Implements: A * 10^( B * temp / (C + temp))"""
        return self._A * (10 ** ((self._B * temp) / (self._C + temp)))
        
    def calc_tdew(self, vp) : 
        """Implements: C / ( (B/log10(vp/A)) - 1 )"""
        return self._C / (self._B/np.log10(vp/self._A)-1)
        
class BuckApproximation (SaturationVapor) : 
    """The Buck approximation has a form that is nearly as computationally 
    efficient as the Magnus approximation for the temperature to vapor pressure
    calculation. However, the inverse is not efficient at all."""
    def __init__(self, A, B, C, D) :
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        
    def calc_vp(self, tempc) : 
        """Implements A*exp(t*(B - t/C)/(D + t))"""
        return self._A * np.exp(tempc * (self._B - tempc/self._C) /(tempc + self._D))
        
    def calc_tdew(self, vp) : 
        """Sympy algebraic solver gives the expression for 't' as: 
           B*C/2 - 
           sqrt(C)*sqrt(B**2*C + 2*B*C*log(A) - 2*B*C*log(svp) + 
                        C*log(A)**2 - 2*C*log(A)*log(svp) + 
                        C*log(svp)**2 + 4*D*log(A) - 
                        4*D*log(svp))/2 + 
           C*log(A)/2 - C*log(svp)/2"""

        t = self._B * self._C / 2
        t -= np.sqrt(self._C) * np.sqrt((self._B**2)*self._C + 
                2*self._B*self._C*(np.log(self._A) - np.log(vp)) +
                (self._C*np.log(self._A)**2) -
                2*self._C*np.log(self._A)*np.log(vp) +
                self._C*np.log(vp)**2 + 
                4*self._D*np.log(self._A) - 
                4*self._D*np.log(vp)) / 2
        t += self._C*np.log(self._A) / 2
        t -= self._C*np.log(vp)/2
        return t
        
vp_calcs={ "AERK" : MagnusExp(610.94, 17.625, 243.04, [-40,50]),
           "AEDK" : MagnusExp(611.02, 17.621, 242.97, [-40,50]),
           "AEDG" : MagnusExp(611.05, 17.546, 241.81, [-40,50]),
           "AEDW" : MagnusExp(611.28, 17.610, 242.89, [-40,50]),
           "AEDS" : MagnusExp(611.52, 17.616, 242.91, [-40,50]),
           "AERG" : MagnusExp(610.72, 17.578, 242.25, [-40,50]),
           "AERW" : MagnusExp(610.85, 17.654, 243.49, [-40,50]),
           "AERS" : MagnusExp(611.07, 17.660, 243.51, [-40,50]),
           "AT85" : MagnusExp(610.70, 17.38, 239.0, [-40,50]),
           "TE30" : Magnus10(611, 7.5, 237.3, [-40,50]),
           "MA67" : Magnus10(610.78, 7.63, 241.9, [-40,50]),
           "BU81" : MagnusExp(611.21, 17.502, 240.97, [-40,50]),
           "AL88" : Magnus10(610.7, 7.665, 243.33, [-40, 50]),
           "SA90" : MagnusExp(611.2, 17.62, 243.12, [-40, 50]),
           "BU-2" : BuckApproximation(611.21, 18.729, 227.3, 257.87)}
"""vp_calcs is a collection of predefined approximate saturation vapor pressure 
calculators. The names in the dictionary and the coefficients for the approximations
come from Table 1 of Alduchov, 1996. See Table 2 in that same paper for 
estimates of accuracy and relative error over the temperature range -40C to 50C. 
The recommendation of the paper is to use the "AERK" version.

Coefficients relate saturation vapor pressure in Pascals to dewpoint temperature
in degrees C.

    Alduchov, Oleg A., and Robert E. Eskridge. “Improved Magnus Form 
      Approximation of Saturation Vapor Pressure.” Journal of Applied 
      Meteorology 35, no. 4 (April 1, 1996): 601–9. 
      doi:10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2.
"""

default = vp_calcs["AERK"]
           
