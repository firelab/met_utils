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
          