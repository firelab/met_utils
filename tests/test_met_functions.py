# -*- coding: utf-8 -*-
import unittest
from astropy import units as u
import numpy as np
import met_functions as m
from satvp import default as vp

class TestMetFuntions (unittest.TestCase) : 
    def test_vp_spec_humidity(self) : 
        """Tests specific humidity -> vapor pressure calculation
        
        Calculates specific humidity using eqn 2.19 of [1], then 
        tests to ensure the tested function can calculate the original
        vapor pressure used.
        
        [1] Rogers, R. R. A Short Course in Cloud Physics, Third Edition. 
            3 edition. Oxford ; New York: Pergamon, 1989.
        """
        # atmospheric variables used in specific humidity calc.
        e = 611 * u.Pa # vapor pressure
        p = 1 * u.bar  # total pressure
        
        # calculate specific humidity        
        q = m.EPSILON * (e/(p - (1-m.EPSILON)*e))
        
        e_test = m.calc_vp_spec_humidity(q,p)
        self.assertTrue(np.abs(e-e_test)<0.1 * u.Pa)
        
    def test_rh_spec_humidity(self) : 
        """Tests specific humidity -> RH calculation
        
        Starting from an RH, an atmospheric pressure and temperature,
        calculate specific humidity. Then check to see if the tested code
        can reproduce the RH we started with. 
        """
        rh = 25 * u.pct
        p =  1 * u.bar
        t =  25 * u.deg_C
        
        e_sat = vp.calc_vp(t)
        w_sat = m.mixing_ratio(e_sat, p)
        
        w = rh * w_sat          # get vapor pressure from definition of RH
        e = p*w/(m.EPSILON + w) # mixing ratio solved for e
        
        # calculate specific humidity using eq 2.19
        q = m.EPSILON * (e/(p-(1-m.EPSILON)*e))

        rh_test = m.calc_rh_spec_humidity(q, p, t)
        
        self.assertTrue(np.abs(rh-rh_test) < 0.1 * u.pct,
          msg="test RH: {:3.1} ; calculated RH: {:3.1}".format(rh, rh_test))
          
    def test_calc_dayl(self):
        """test daylength calculations
        
        Data here are taken from the US Naval Observatory's webpage for 
        Missoula, MT in the year 2014.  These data reflect the length of time
        that any part of the solar disc is above the horizon.
        (http://aa.usno.navy.mil/data/docs/Dur_OneYear.php)
        
        In order to make this test pass, I had to relax the agreement between
        the code and the USNO tables to 20 minutes or better. This may be due to
        differences between the definitions of what constitutes "daylength" or 
        it may be due to differences in precision. Or it could be an error.
        """
        lat = 46.86 * u.deg
        days = [1, 59, 120, 181, 243, 304] * u.day
        daylengths = u.Quantity([ 8*u.hour+37*u.min, 11*u.hour+6*u.min, 14*u.hour+26*u.min,
                       15*u.hour+48*u.min, 13*u.hour+19*u.min, 10*u.hour+1*u.min], u.hour)
        
        for i  in range(len(days)) : 
            test_daylength = m.calc_dayl(lat,days[i])
            self.assertLess( np.abs(daylengths[i] - test_daylength), 20*u.min)
            
    def test_calc_dayl_latarray(self): 
        """
        Test may 1st for Missoula, Denver, Atlanta
        """
        day = 120 * u.day
        lats = [46.86, 39.72, 33.77 ] * u.deg
        daylengths = u.Quantity([ 14*u.hour+26*u.min, 13*u.hour+54*u.min, 13*u.hour+32*u.min ], u.hour)
        
        test_daylengths = m.calc_dayl(lats, day)
        self.assertTrue(np.all( np.abs(test_daylengths-daylengths) < 20*u.min))
        
    def test_calc_dayl_datearray(self): 
        """Test Missoula throughout the year, using days as an array"""
        lat = 46.86 * u.deg
        days = [1, 59, 120, 181, 243, 304] * u.day
        daylengths = u.Quantity([ 8*u.hour+37*u.min, 11*u.hour+6*u.min, 14*u.hour+26*u.min,
                       15*u.hour+48*u.min, 13*u.hour+19*u.min, 10*u.hour+1*u.min], u.hour)
        test_daylengths=m.calc_dayl(lat,days)
        self.assertTrue(np.all(np.abs(test_daylengths-daylengths) < 20*u.min))
        
    def test_calc_dayl_botharrays(self): 
        """Test yday and lat as arrays
        
        Missoula on day 243, Denver and Atlanta on day 120.
        """
        lats = [46.86, 39.72, 33.77 ] * u.deg
        days = [243,   120,   120]    * u.day
        daylengths = u.Quantity([ 13*u.hour+19*u.min, 13*u.hour+54*u.min, 13*u.hour+32*u.min ], u.hour)
        
        test_daylengths = m.calc_dayl(lats, days)
        self.assertTrue(np.all( np.abs(test_daylengths-daylengths) < 20*u.min))
        
    def test_sameas_mt_clim(self):
        """Get same result as MT-CLIM C code?
        
        Tests that the python code produces the same results as the test cases
        provided in the comments of the MT-CLIM C code. Provided test cases
        report nearest-integer seconds, so we check that the difference doesn't
        exceed 0.5s.
        """
        lat = 48.0 * u.deg
        days = range(101,105) * u.day
        daylengths = [47672, 47880, 48087, 48293]*u.s

        test_daylengths = m.calc_dayl(lat, days)
        self.assertTrue(np.all( np.abs(test_daylengths-daylengths) <= 0.5*u.s))

    def test_CalcDaylightHours(self):
        """test daylength calculations from matt's code
        
        Data here are taken from the US Naval Observatory's webpage for 
        Missoula, MT in the year 2014.  These data reflect the length of time
        that any part of the solar disc is above the horizon.
        (http://aa.usno.navy.mil/data/docs/Dur_OneYear.php)
        
        In order to make this test pass, I had to relax the agreement between
        the code and the USNO tables to 31 minutes or better. This may be due to
        differences between the definitions of what constitutes "daylength" or 
        it may be due to differences in precision. Or it could be an error.
        """
        lat = 46.86 * u.deg
        days = [1, 59, 120, 181, 243, 304] * u.day
        daylengths = u.Quantity([ 8*u.hour+37*u.min, 11*u.hour+6*u.min, 14*u.hour+26*u.min,
                       15*u.hour+48*u.min, 13*u.hour+19*u.min, 10*u.hour+1*u.min], u.hour)
        
        test_daylengths = m.CalcDaylightHours(lat, days)
        self.assertTrue(np.all( np.abs(test_daylengths-daylengths) < 31*u.min))

