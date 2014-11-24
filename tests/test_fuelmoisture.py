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
        self.assertEqual(emc_c.unit, u.pct)
        
    def test_eqmc_array(self) : 
        """ensures that eqmc calculator works with arrays."""
        rh = np.arange(20,80, 5) * u.pct
        tf = np.arange(80,20, -5) * iu.deg_F
        self.assertEqual(rh.size, tf.size)
        
        emc = fm.eqmc( tf, rh)
        self.assertEqual(emc.size, rh.size)
        
    def test_eqmc_bar_units(self) : 
        """checks that eqmc_bar returns expected values"""
        rh_min = 15*u.pct
        rh_max = 60*u.pct
        t_min  = 60*iu.deg_F
        t_max  = 85*iu.deg_F
        daylength = 15*u.hour
        testval  = fm.eqmc_bar(daylength, t_max, t_min, rh_max, rh_min)
        
        emax = fm.eqmc(t_min,rh_max)
        emin = fm.eqmc(t_max,rh_min)
        refval = (daylength*emin + (1*u.day-daylength)*emax)/(1*u.day)
        
        self.assertEqual(testval,refval)
        self.assertEqual(testval.unit, u.pct)
        
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
        
    def test_precip_sub_day(self):
        """Tests that interpretation of precip works"""
        
        # testing certain scalars
        self.assertLess(fm.precip_duration_sub_day([0]/u.day, 4 / u.day) - 0 * u.hour, 
            1e-5*u.hour, "Failed to produce zero precip!")
        self.assertLess(fm.precip_duration_sub_day([1]/u.day, 4 / u.day) - 1 * u.hour, 
            1e-5*u.hour, "Single rain observation should be 1 hour rain")
        interpval = (1 + (3-1)*(24-1)/(4-1.))
        self.assertLess(fm.precip_duration_sub_day([3]/u.day, 4 / u.day) - interpval * u.hour,
            1e-5*u.hour, "More than one rain observation should be interpolated") 
            
        # testing array
        self.assertTrue(np.all(fm.precip_duration_sub_day( [0,1,3]/u.day, 4/u.day) - 
                               [0,1,interpval] * u.hour < 1.e-5 * u.hour),
                               "Array data type failed")
                               
        # testing the "no units" case
        self.assertLess(fm.precip_duration_sub_day(np.array([1]), 4) - 1 * u.hour, 
            1e-5*u.hour, "Single rain observation should be 1 hour rain")
                               
class TestFMCalculators (unittest.TestCase) : 
    def setUp(self) : 
        # 14 days by 8 land points
        self.land_points = 8
        self.shape = (14,self.land_points)
        self.time_axis = 0 
        self.precip_duration = np.zeros( (self.land_points,) ) * u.hour
        self.eqmc_bar = np.ones( (self.land_points,)) * 30 * u.pct
        
    def test_create_fm1000(self) : 
        """can we make a 1000 hr fm object"""
        x = fm.ThousandHourFM(self.shape, self.time_axis) 
    
    def test_create_fm100(self) : 
        """can we make a 100 hr fm object"""
        x = fm.HundredHourFM(self.shape, self.time_axis)
        
    def compute_1000_hr(self) : 
        """basic 1000-hr fuel moisture calculation, no precip"""
        x = fm.ThousandHourFM(self.shape, self.time_axis)
        testval = x.compute(self.eqmc_bar, self.precip_duration) 
        self.assertTrue(hasattr(testval),'unit')
        self.assertEqual(testval.unit, u.pct)
        self.assertEqual(len(testval.shape),1)
        self.assertEqual(testval.shape[0], self.land_points)
        
        boundary_bar = ((15*u.pct * 6) + 30*u.pct) / 7
        refval = 15*u.pct + (boundary_bar - 15*u.pct) * .3068
        self.assertTrue(np.all(refval == testval))
        
        
    def compute_100_hr(self): 
        """basic 100-hr fuel moisture calculation, no precip"""
        x = fm.HundredHourFM(self.shape, self.time_axis)
        testval = x.compute(self.eqmc_bar, self.precip_duration) 
        self.assertTrue(hasattr(testval),'unit')
        self.assertEqual(testval.unit, u.pct)
        self.assertEqual(len(testval.shape),1)
        self.assertEqual(testval.shape[0], self.land_points)
        
        refval = 20*u.pct + 0.3156 * (self.eqmc_bar - 20*u.pct)
        self.assertTrue(np.all(refval == testval))
        
        
        
    
                

        