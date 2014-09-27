# -*- coding: utf-8 -*-
"""
Functions and classes to calculate fuel moisture.

This module contains code to calculate fuel moisture given typical data sources
available from meterological and climate models. The code is written with 
two objectives: 
    
    1. Encapsulate the various calculations in appropriate objects.
    2. Efficiently store state.
    
These objectives are in tension, as my use case is to process met data 
by iterating over the time steps, loading a 2D grid of my simulation domain
at each timestep. Fully encapsulated calculation objects would store state 
internally, whereas efficient calculation objects would maintain their state
(particularly history) in a numpy ndarray object.

This speaks to the need for a calculation manager object, which can provide
external state management for calculation objects which deal with one calculation
at a time.
"""
from astropy import units as u
from astropy.units import imperial as iu
import image_chain as ic

## \fn eqmc (self,Temp,RH)
## \brief Calculate the equilibrium moisture content
## \param Temp Temperature (deg F)
def eqmc (Temp,RH):
    """Calculate the equilibrium moisture content.
    
    Parameters
    ----------
    Temp : array : degF
    RH   : array : pct
    
    Returns
    -------
    eqmc : array : pct
    
    This function does not maintain state between calls, so it does not need to
    be inside a class. EMC equations are presented in Appendix C of [1], as
    equations C-1, C-2, and C-3. Note that the RH ranges which select which 
    equation to use have been fixed so they are not overlapping.
    
    [1] Bradshaw, Larry S., John E. Deeming, Robert E. Burgan, and Jack D. Cohen. 
    1984. The 1978 National Fire-Danger Rating System: Technical Documentation. 
    http://www.treesearch.fs.fed.us/pubs/29615.    
    """

    if (Temp.unit != iu.deg_F) : 
        Temp = Temp.to(iu.deg_F, u.temperature())
    if(RH > 50*u.pct):
        return  (21.0606*u.pct + (0.005565 * 1/u.pct) * RH**2 - 
                 ((0.00035 * 1/iu.deg_F) * RH * Temp) - 0.483199 * RH)
    if(RH > 10*u.pct) and (RH <= 50*u.pct):
        return (2.22749*u.pct + 0.160107 * RH - (0.014784 * u.pct/iu.deg_F) * Temp)
    else:
        return (0.03229*u.pct + 0.281073 * RH - (0.000578 * 1/iu.deg_F) * RH * Temp)	

## \fn oneten (Temp, RH, SOW)
## \brief Calculate the one and ten hour moisture contents
## \param Temp Temperature (deg F)
## \param RH Relative Humidity (%)
## \param SOW State of the Weather (0-9)
def oneten_nfdrs (Temp, RH, SOW):
    """One and ten hour fuel moisture computation using 1978 NFDRS method.
    
    This function adjusts standard exposed instrument readings (for temperature
    and RH) from 4.5 feet off the ground in a shelter to "fuel level" as per 
    Table 6 in [1]. "State of the weather" code definitions: 
        SOW = 0 : Clear sky (0.0-0.1 cloud cover)
        SOW = 1 : Scattered (0.1-0.5 cloud cover)
        SOW = 2 : Broken    (0.6-0.9 cloud cover)
        anything else : Overcast (0.9-1.0 cloud cover)
    
    [1] Bradshaw, Larry S., John E. Deeming, Robert E. Burgan, and Jack D. Cohen. 
    1984. The 1978 National Fire-Danger Rating System: Technical Documentation. 
    http://www.treesearch.fs.fed.us/pubs/29615.
    
    Parameters
    ----------
    Temp : array : degF
    RH   : array : pct
    SOW  : array : dimensionless
    
    Returns
    -------
    fm : array : pct
        Tuple containing the (1-hr, 10-hr) fuel moistures.
    """
    
    # Determine the temperature and rh factors to adjust for fuel temperatures
    if SOW == 0:
        tfact = 25.0 * iu.deg_F
        hfact = 0.75
    if SOW == 1:
        tfact = 19.0 * iu.deg_F
        hfact = 0.83
    if SOW == 2:
        tfact = 12.0 * iu.deg_F
        hfact = 0.92
    else:
        tfact = 5.0 * iu.deg_F
        hfact = 1.0
  	
    if Temp.unit != iu.deg_F : 
       Temp = Temp.to(iu.deg_F, u.temperature())	
    emc = eqmc(tfact + Temp, hfact * RH)    
    return (1.03*emc, 1.28*emc)
    
def oneten_ofdm(temp, rh, srad, fm_100) : 
    """one and ten hour fuel moisture using the Oklahoma fire danger model
    
    Calculates the one and ten hour fuel moistures using the methodology in [1].
    This method does not require knowledge of the "state of the weather", and 
    may be easier to use with simulated met data.
    
    Parameters
    ----------
    temp : array : deg_C
        1.5 m temperature observation
    rh   : array : pct
        1.5 m relative humidity observation
    srad : array : W / m**2
        1.5 m solar radiation observation from Li-Cor LI-200s pyranometer 
        (bandpass: 400-1100nm)
    fm_100 : array : pct
        calculated 100-hr fuel moisture (dry basis)
        
    Returns
    -------
    fm : array : pct
        tuple containing 1 and 10 hour % fuel moistures
    
    
    [1] Carlson, J. D., Robert E. Burgan, David M. Engle, and 
        Justin R. Greenfield. 2002. “The Oklahoma Fire Danger Model: An 
        Operational Tool for Mesoscale Fire Danger Rating in Oklahoma.” 
        International Journal of Wildland Fire 11 (4): 183–91.
    """
    
    fuel_temp = (srad/1353)*(13.9 * u.m**2 * u.deg_C /u.W) + temp
    fuel_rh   = (1 - (0.25*u.m**2/u.W)*(srad/1353))*rh
    
    fm_basis = 0.8 * emc(fuel_temp, fuel_rh) 
    fm_10 = fm_basis + 0.2 * fm_100
    fm_1  = fm_basis + 0.2 * fm_10
    
    return (fm_1, fm_10)

	## \fn hundredthous (self, Temp,RH,MaxTemp,MaxRH,MinTemp, MinRH,Julian,PrecipDur)
	## \brief Calculate the hundred and thousand hour moisture contents
	## \param Temp Temperature (deg F)
	## \param RH Relative Humidity (%)
	## \param MaxTemp 24-hour Maximum Temperature (deg F)
	## \param MaxRH 24-hour Maximum Relative Humidity (%)
	## \param MinTemp 24-hour Minimum Temperature (deg F)
	## \param MinRH 24-hour Minimum Relative Humidity (%)
	## \param Julian Julian Year Day (1-366)
	## \param 24-hour Precipitation Duration (hours)
#	def hundredthous (self, Temp,RH,MaxTemp,MaxRH,MinTemp, MinRH,Julian,PrecipDur):
#		emcMin = 0.0
#		emcMax = 0.0
#		emcBar = 0.0
#		bndryH = 0.0
#		bndryT = 0.0
#		bndryBar = 0.0
#		Daylight = 0.0
#		ambvp = 0.0
#			  
#		Daylight = self.CalcDaylightHours( Julian)
#	   	emcMin = eqmc (MaxTemp, MinRH)
#		emcMax = eqmc (MinTemp, MaxRH)
#	   	emcBar = (Daylight * emcMin + (24 - Daylight) * emcMax) / 24
#	   	
#	   	# eqn 36
#	   	# bndryH == boundary 100-hour
#	   	bndryH = ((24.0 - PrecipDur) * emcBar + PrecipDur * (.5 * PrecipDur + 41)) / 24
#	   	
#	   	# Y100 == 100-hour moisture content from previous iteration (day)
#	   	# eq 37, but 0.3156 is not the specified constant
#		self.MC100 = self.Y100 + .3156 * (bndryH - self.Y100);
#		
#		# eqn 41
#		# bndryT == boundary 1000-hour
#		bndryT = ((24.0 - PrecipDur) * emcBar + PrecipDur * (2.7 * PrecipDur + 76)) / 24
#
#		# Note this can be rewritten using Python push and pop
#		for i in range (0,6):
#			self.HistBndryT[i] = self.HistBndryT[i+1]
#			bndryBar = bndryBar + self.HistBndryT[i]
#	   	
#	   	self.HistBndryT[6] = bndryT
#	   	# eq 40
#	   	bndryBar = (bndryBar + bndryT) / 7
#	   	
#	   	# eq 42, but 0.3068 is not the specified constant
#	   	self.MC1000 = self.Hist1000[0] + (bndryBar - self.Hist1000[0]) * .3068
#	   	
#	   	# maintain history buffer for 1000 hour moistures
#	   	# the above eqn (eqn 42) uses the 7-day-prior initial value.
#	   	for i in range (0,6):
#	   		self.Hist1000[i] = self.Hist1000[i+1]
#		
#	   	self.Hist1000[6] = self.MC1000
#		self.Y100 = self.MC100
#		self.Y1000 = self.MC1000
