import unittest
from astropy import units as u
from astropy.units import imperial as iu
import fuelmoisture as fm
import numpy as np

class TestFuelMoisture(unittest.TestCase) : 
    def test_eqmc_units(self):
        """Checks for proper temperature units conversion"""
        rh = 50*u.pct
        tc = 30*u.deg_C
        tf = tc.to(iu.deg_F, u.temperature())
        
        emc_c = fm.eqmc(tc, rh)
        emc_f = fm.eqmc(tf, rh)
        
        self.assertLess( np.abs(emc_c - emc_f), 0.1*u.pct)
        
    def test_oneten_units(self): 
        """Checks for proper temperature units conversion"""
        rh = 50*u.pct
        tc = 30*u.deg_C
        tf = tc.to(iu.deg_F, u.temperature())
        
        ot_c = fm.oneten_nfdrs(tc,rh,0)
        ot_f = fm.oneten_nfdrs(tf,rh,0)
        
        self.assertLess( np.abs(ot_c[0]-ot_f[0]), 0.1*u.pct)
        
    def test_oneten_values(self):
        """Tests that the ratio of one and ten hour fuel moistures are
        as expected."""
        rh = 50*u.pct
        tc = 30*u.deg_C

        ot = fm.oneten_nfdrs(tc, rh, 0)
        expected_ratio = 1.03/1.28
        self.assertLess(np.abs(ot[0]/ot[1]-expected_ratio), 1e-5)