# -*- coding: utf-8 -*-
"""Classes and concrete calculators to express the relationships between 
saturation vapor pressue and dewpoint temperature. These are peer-reviewed.
If you don't care which one is best, just use "default", and use degrees C and 
Pascals."""

from abc import ABCMeta, abstractmethod
import numpy as np
from astropy import units as u


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
        """ Instantiates a mangus approximation class.
            
            Parameters
            ----------
            A : scalar : units = Pa
            B : scalar : units = dimensionless
            C : scalar : units = degC
            range : array : units = degC
        """
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
        """Implements: A * e^( B * temp / (C + temp))
        
           Parameters
           ----------
           temp : array : units = degC
           
           Returns
           -------
           c : array : units = Pa
        """
        return self._A * np.exp((self._B * temp) / (self._C + temp))
        
    def calc_tdew(self, vp) :
        """Implements: C / ( (B/ln(vp/A)) - 1 )
        
           Parameters
           ----------
           vp : array : units = Pa
           
           Returns
           -------
           t : array : units = degC
        """
        return self._C / (self._B/np.log(vp/self._A)-1)
        
class Magnus10 (MagnusApproximation) : 
    """Concrete class which implements a magnus approximation using log base 10
    and powers of 10"""
    
    def calc_vp(self, temp) :
        """Implements: A * 10^( B * temp / (C + temp))
        
           Parameters
           ----------
           temp : array : units = degC
           
           Returns
           -------
           c : array : units = Pa
        """
        return self._A * (10 ** ((self._B * temp) / (self._C + temp)))
        
    def calc_tdew(self, vp) : 
        """Implements: C / ( (B/log10(vp/A)) - 1 )
        
           Parameters
           ----------
           vp : array : units = Pa
           
           Returns
           -------
           t : array : units = degC
        """
        return self._C / (self._B/np.log10(vp/self._A)-1)
        
class BuckApproximation (SaturationVapor) : 
    """The Buck approximation has a form that is nearly as computationally 
    efficient as the Magnus approximation for the temperature to vapor pressure
    calculation. However, the inverse is not efficient at all. In addition, the 
    mathematical expression for the inverse has us taking the log of 
    quantities with units, which is not allowed by astropy."""
    def __init__(self, A, B, C, D) :
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        
    def calc_vp(self, tempc) : 
        """Implements A*exp(t*(B - t/C)/(D + t))
        
           Parameters
           ----------
           tempc : array : units = degC
           
           Returns
           -------
           c : array : units = Pa
        """
        return self._A * np.exp(tempc * (self._B - tempc/self._C) /(tempc + self._D))
    
    def calc_tdew(self, vp) : 
        """Sympy algebraic solver gives the expression for 't' as: 
           B*C/2 - 
           sqrt(C)*sqrt(B**2*C + 2*B*C*log(A) - 2*B*C*log(svp) + 
                        C*log(A)**2 - 2*C*log(A)*log(svp) + 
                        C*log(svp)**2 + 4*D*log(A) - 
                        4*D*log(svp))/2 + 
           C*log(A)/2 - C*log(svp)/2
           
           Parameters
           ----------
           vp : array : units = Pa
           
           Returns
           -------
           t : array : units = degC
           """

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

nd = u.dimensionless_unscaled        
vp_calcs={ "AERK" : MagnusExp(610.94*u.Pa, 17.625*nd, 243.04*u.deg_C, np.array([-40,50])*u.deg_C),
           "AEDK" : MagnusExp(611.02*u.Pa, 17.621*nd, 242.97*u.deg_C, np.array([-40,50])*u.deg_C),
           "AEDG" : MagnusExp(611.05*u.Pa, 17.546*nd, 241.81*u.deg_C, np.array([-40,50])*u.deg_C),
           "AEDW" : MagnusExp(611.28*u.Pa, 17.610*nd, 242.89*u.deg_C, np.array([-40,50])*u.deg_C),
           "AEDS" : MagnusExp(611.52*u.Pa, 17.616*nd, 242.91*u.deg_C, np.array([-40,50])*u.deg_C),
           "AERG" : MagnusExp(610.72*u.Pa, 17.578*nd, 242.25*u.deg_C, np.array([-40,50])*u.deg_C),
           "AERW" : MagnusExp(610.85*u.Pa, 17.654*nd, 243.49*u.deg_C, np.array([-40,50])*u.deg_C),
           "AERS" : MagnusExp(611.07*u.Pa, 17.660*nd, 243.51*u.deg_C, np.array([-40,50])*u.deg_C),
           "AT85" : MagnusExp(610.70*u.Pa, 17.38*nd, 239.0*u.deg_C, np.array([-40,50])*u.deg_C),
           "TE30" : Magnus10(611*u.Pa, 7.5*nd, 237.3*u.deg_C, np.array([-40,50])*u.deg_C),
           "MA67" : Magnus10(610.78*u.Pa, 7.63*nd, 241.9*u.deg_C, np.array([-40,50])*u.deg_C),
           "BU81" : MagnusExp(611.21*u.Pa, 17.502*nd, 240.97*u.deg_C, np.array([-40,50])*u.deg_C),
           "AL88" : Magnus10(610.7*u.Pa, 7.665*nd, 243.33*u.deg_C, np.array([-40,50])*u.deg_C),
           "SA90" : MagnusExp(611.2*u.Pa, 17.62*nd, 243.12*u.deg_C, np.array([-40,50])*u.deg_C)}
           #"BU-2" : BuckApproximation(611.21*u.Pa, 18.729*nd, 227.3*u.deg_C, 257.87*u.deg_C)}
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
           
