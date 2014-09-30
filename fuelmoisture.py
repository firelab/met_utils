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
import numpy as np
import numpy.ma as ma

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
        
    The scaling factors applied to equilibrium moisture content to obtain 
    one and ten hour fuel moistures (e.g., in the return statement) have the 
    following caveat:
        
        This model works well for early afternoons in strong
        continental areas at the approximate latitude of
        Nebraska in the late summer. It tends to underpredict
        stick readings under other conditions.
        
    See [1] for further details.
    
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
    
    Calculates the one and ten hour fuel moistures using the methodology in [2].
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
    
    
    [2] Carlson, J. D., Robert E. Burgan, David M. Engle, and 
        Justin R. Greenfield. 2002. “The Oklahoma Fire Danger Model: An 
        Operational Tool for Mesoscale Fire Danger Rating in Oklahoma.” 
        International Journal of Wildland Fire 11 (4): 183–91.
    """
    
    fuel_temp = (srad/1353)*(13.9 * u.m**2 * u.deg_C /u.W) + temp
    fuel_rh   = (1 - (0.25*u.m**2/u.W)*(srad/1353))*rh
    
    fm_basis = 0.8 * eqmc(fuel_temp, fuel_rh) 
    fm_10 = fm_basis + 0.2 * fm_100
    fm_1  = fm_basis + 0.2 * fm_10
    
    return (fm_1, fm_10)


def precip_duration_sub_day(rainy_periods, daily_obs) : 
    """Estimates precip duration from multiple daily observations
    
    High temporal resolution data sources can provide better guesses
    at the length of a rainfall event than the "default" conversion of 
    amount to duration based on a climate class. The algorithm embodied 
    in this function is based on a conversation with Matt Jolly 9/29/2014.
    Matt had the idea, the errors are likely mine.
    
    Algorithm: 
        1. If none of the daily measurement periods had rain, precip duration is zero.
        2. If there's rain in only one measurement period, assume it rained 
            for only an hour.
        3. Above one measurement period, linearly interpolate between 
            (1 period, 1 hour) and (N periods, 24 hours) where N is the number
            of daily observations.
            
    Parameters
    ----------
    rainy_periods : array: day**(-1)
        number of rainy periods in the last day
    daily_obs : scalar : day**(-1)
        total number of observations in a day
        
    Returns
    -------
    estimate of the precipitation duration
    """
    precip_interp_x = np.array([0,1,(daily_obs/(1/u.day))])  # / u.day
    precip_interp_y = np.array([0.,1.,24.]) # * u.hour
    return np.interp(rainy_periods, precip_interp_x, precip_interp_y) * u.hour
    
    
def precip_duration(amt, cc) : 
    """calculates the precipitation duration from the amount
    
    Given the amount of precipitation over the last day, calculates the 
    duration of the precipitation. This calculation depends on the 
    climate class definitions in the 1978 NFDRS, as the average rainfall
    rate is taken to have different values. 
    
    This function implements eqn 39 in [1]
    
    [1] Bradshaw, Larry S., John E. Deeming, Robert E. Burgan, and 
        Jack D. Cohen. 1984. The 1978 National Fire-Danger Rating System: 
        Technical Documentation. http://www.treesearch.fs.fed.us/pubs/29615.
    
    Parameters
    ----------
    amt : array : inch
        total amount of rainfall in the last day
    cc  : array : dimensionless
        numerical indication of climate class (1-4)
        
    Returns
    -------
    duration : array : hours
    """
    rate = None
    if cc in [1,2] : 
        rate = 0.25 * iu.inch / u.hour
    elif cc in [3,4] : 
        rate = 0.05 * iu.inch / u.hour
    
    return (amt + 0.02 * iu.inch) / rate
    
    
def eqmc_bar(daylight, t_max, t_min, rh_max, rh_min):
    """calculates a weighted average equilibrium moisture content
    
    Implements equation 38 in [1]. The result is an effective equilibrium 
    moisture content which can be considered to apply to the whole day. The 
    max and min values for temp and rh apply to a single day, for which this
    function calculates the effective equilibrium moisture content.
    
    This result is shared between the 100 and 1000 hour fuel moisture 
    calculations. 
    
    [1] Bradshaw, Larry S., John E. Deeming, Robert E. Burgan, and 
        Jack D. Cohen. 1984. The 1978 National Fire-Danger Rating System: 
        Technical Documentation. http://www.treesearch.fs.fed.us/pubs/29615.

    Parameters
    ----------
    daylight : array : hours
        length of time from sunrise to sunset 
        This is a 1D array (column). The value at index "i" applies to the 
        entire "i"th row in the 2D arrays.
    t_max : array : deg_F
        max temperature for the day
    t_min : array : deg_F
        min temperature for the day
    rh_max : array : pct
        maximim relative humidity for the day
    rh_min : array : pct
        minimum relative humidity for the day
    """
    emc_min = eqmc (t_max, rh_min)
    emc_max = eqmc (t_min, rh_max)
    nighttime = 24*u.hour  - daylight
    return (daylight * emc_min + (nighttime * emc_max)) / (24*u.hour)

class HundredHourFM(ic.IndividualStateOperator) : 
    """class to calculate 100 hour fuel moisture
    
    This class calculates 100 hour fuel moisture sequentially
    (one day per call). It remembers the 1000 hour moisture for the previous
    calls, which should correspond to the previous day. It is important to
    provide daily data to this class in time order. 
    """
    
    def _get_history_dtype(self) : 
        """define a record type for the history required by this class
        
        All this class needs to remember is the previous day's fuel moisture
        """
        return np.dtype( [ ('fm_100', np.float64) ] )
        
    def _get_history_depth(self) : 
        return 1
        
    def _init_history(self) :
        self._history[:].fm_100 = 20 * u.pct
        
    def fm_100(self, eqmc_bar, precip_dur) : 
	# eqn 36
	boundary = ((24.0 *u.hour - precip_dur) * eqmc_bar + 
	           precip_dur * ((.5 * u.pct/u.hour) * precip_dur + (41*u.pct))) / (24*u.hour)
	   	
	# eq 37, but 0.3156 is not the specified constant
	prev_fm_100 = self.history[Ellipsis,0].fm_100
	fm_100 = prev_fm_100 + .3156 * (boundary - prev_fm_100);
        
	self._history[Ellipsis,0] = fm_100
	
	self._output_arrays['fm_100'] = fm_100
    
    
class ThousandHourFM (ic.IndividualStateOperator) :
    """class to calculate 1000 hour fuel moisture
    
    This class calculates 1000 hour fuel moisture sequentially
    (one day per call). It remembers the 1000 hour moisture for the past
    seven calls, which should correspond to the last 7 days. It is important to
    provide daily data to this class in time order. 
    """
    
    def _get_history_dtype(self) : 
        """define a record type for the history required by this class
        
        This class only needs to remember boundary calculations and prior 
        fuel moistures.
        """
        return np.dtype( [ ('boundary', np.float64), ('fm_1000', np.float64)] )
        
    def _get_history_depth(self) : 
        """retain one week (7 days/iterations) of history"""
        return 7
        
    def _init_history(self): 
        """initialize the history buffer with default values"""
        # for climate class 1 & 2 : 15% is default, for 3&4 : 23%
        self._history[:].boundary = 15 * u.pct
        
        # no default specified in literature. this is from Matt Jolly's source
        self._history[:].fm_1000  = 15 * u.pct


    ## \fn hundredthous (self, Temp,RH,MaxTemp,MaxRH,MinTemp, MinRH,Julian,PrecipDur)
    ## \brief Calculate the hundred and thousand hour moisture contents
    ## \param 24-hour Precipitation Duration (hours)
    def fm_1000 (self, eqmc_bar, precip_dur):
        """perform the calculation
        
        This function implements the 1978 NFDRS fuel moisture calculations for
        100 and 1000 hour fuels. Equations referenced in the comments come from
        [1]. 
        
        [1] Bradshaw, Larry S., John E. Deeming, Robert E. Burgan, and 
            Jack D. Cohen. 1984. The 1978 National Fire-Danger Rating System: 
            Technical Documentation. http://www.treesearch.fs.fed.us/pubs/29615.
        """
		
        # eqn 41
	# bndryT == boundary 1000-hour
	boundary = ((24.0*u.hour - precip_dur) * eqmc_bar + 
	          precip_dur * ((2.7*u.pct/u.hour) * precip_dur + 76*u.pct)) / (24*u.hour)

	# eq 40 , must mask out oldest boundary record
	# (we want today's boundary val + the previous six values)
        oldest_idx = self._rr_view.get_history_index(6)
        bmask = np.empty( self._history.shape ,dtype = np.bool)
        bmask[:] = False
        bmask[...,oldest_idx] = True
        masked_hist = ma.array(self._history, mask=bmask, copy=False)
        boundary_bar = np.sum(masked_hist[:].boundary, axis=self._rr_view.histdim)
        boundary_bar += boundary # do op in place, no copy
        boundary_bar /= 7        # do op in place, no copy
        
	   	
        # eq 42, but 0.3068 is not the specified constant
        prev_fm_1000 = self._history[...,oldest_idx]
	fm_1000 = prev_fm_1000 + (boundary_bar - prev_fm_1000) * .3068
	
	# cycle the history
	self._rr_view.next()
	
	# store new values in the history
	newest_idx = self._rr_view.get_history_index(0)
	self._history[...,newest_idx].fm_1000 = fm_1000
	self._history[...,newest_idx].boundary = boundary
	
	# store output
	self._output_arrays['fm_1000'] = fm_1000