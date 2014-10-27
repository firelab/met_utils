import unittest
import astropy.units as u
import astropy.coordinates as c
import qty_index as q
import numpy as np


class TestLinearSamplingFunction(unittest.TestCase) : 
    def test_four_samples_per_day(self) : 
        samp_fn = q.LinearSamplingFunction( 4. / u.day)
        hours = np.arange(0,23) * u.hour
        test_ind = samp_fn.get_index(hours)
        self.assertTrue(np.all(test_ind == np.trunc(hours/(6*u.hour))))
        
class TestLongitudeSamplingFunction(unittest.TestCase) : 
    def test_noon(self) : 
        samp_fn = q.LongitudeSamplingFunction(4./u.day , 12*u.hour)
        lons = np.arange(-180, 180, 30) * u.deg
        test_ind = samp_fn.get_index(lons)
        delta_hours = (c.Angle(180*u.deg - lons).wrap_at(360*u.deg) / (15 *(u.deg/u.hour)))
        ind = np.trunc(delta_hours / (6*u.hour))
        self.assertTrue(np.all(test_ind == ind))
        
    def test_1400(self) : 
        samp_fn = q.LongitudeSamplingFunction(4./u.day , 14*u.hour)
        lons = np.arange(-180, 180, 30) * u.deg
        test_ind = samp_fn.get_index(lons)
        ang_1400 = (14./24.) * 360*u.deg
        delta_hours = (c.Angle(ang_1400 - lons).wrap_at(360*u.deg) / (15 *(u.deg/u.hour)))
        ind = np.trunc(delta_hours / (6*u.hour))
        self.assertTrue(np.all(test_ind == ind))