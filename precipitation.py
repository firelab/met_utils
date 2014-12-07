"""Defines conversion factors for precipitation.

This allows conversion between mm rainfall and kg/m^2. The conversion relies on 
the mass density of water being 1 kg / L. Also, every mm of height over a 1 m^2 
area is a volume of 1 L.
"""
import astropy.units as u


def precipitation() : 
    return [ (u.kg/(u.m**2), u.mm, lambda x: x, lambda x: x)]