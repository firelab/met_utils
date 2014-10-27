from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as u
import astropy.coordinates as c

class SamplingFunction (object) : 
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_index(self, unit_val) : 
        pass
        
class LinearSamplingFunction ( SamplingFunction ) :
    def __init__(self, scale, offset=0) :
        self.scale = scale
        self.offset = offset
        
    def get_index(self, unit_val) : 
        return np.trunc(unit_val*self.scale + self.offset).astype(np.int)
        
class LongitudeSamplingFunction (SamplingFunction) : 
    E_ROT = 360*u.deg/(1*u.day)
    """earth angular velocity around own axis (approximate)"""
    
    def __init__(self, daily_samples, time_of_day) :
        self.daily_samples = daily_samples
        self.time_of_day = time_of_day
        self.time_angle = c.Angle( (time_of_day /(24*u.hour)) * 360 * u.deg)
        sample_interval = (1./daily_samples) * u.day
        self.time_to_sample = LinearSamplingFunction( 1/sample_interval )
        
    def get_index(self, unit_val) : 
        """converts longitude to sample"""
        lon_angle = c.Angle((-unit_val) + self.time_angle).wrap_at(360*u.deg) 
        return self.time_to_sample.get_index(lon_angle/self.E_ROT)
        