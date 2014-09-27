import unittest
from astropy import units as u
import numpy as np
import gsi


class TestNormalize(unittest.TestCase) : 
    def setUp(self) : 
        self.xform = gsi.Normalize(200.,300.)
        
    def test_middle(self) : 
        """a value about in the middle"""
        self.assertAlmostEqual(0.5, self.xform.scale(250))
        
    def test_proportional(self) : 
        """25% of the way from the lower bound"""
        self.assertAlmostEqual(0.25, self.xform.scale(225))
        
    def test_low(self) : 
        """a value off the low end of the scale"""
        self.assertAlmostEqual(0, self.xform.scale(100))
        
    def test_high(self) : 
        """a value off the top end of the scale"""
        self.assertAlmostEqual(1, self.xform.scale(5000))
        
    def test_array(self) : 
        """give it an array instead of a scalar"""
        x = np.array( [200, 250, 300] )
        x_prime = self.xform.scale(x)
        self.assertEqual(len(x_prime), 3)
        self.assertAlmostEqual(0, x_prime[0])
        self.assertAlmostEqual(0.5, x_prime[1])
        self.assertAlmostEqual(1, x_prime[2])
        
class TestGSI(unittest.TestCase) : 
    def setUp(self) :
        self.gsi = gsi.GSI()
        self.xf_tmin = self.gsi.xf_tmin
        self.xf_vpd  = self.gsi.xf_vpd
        self.xf_photo= self.gsi.xf_photo
        
    def test_photoperiod_low(self) : 
        """a value off the low end of the scale"""
        self.assertLess(self.xf_photo.scale(9*u.hour) - 0, 1e-5)
        
    def test_photoperiod_middle(self) : 
        """a value about in the middle"""
        self.assertLess(self.xf_photo.scale(10.5*u.hour) - 0.5, 1e-5)
        
    def test_photoperiod_high(self) : 
        """a value off the top end of the scale"""
        self.assertLess(self.xf_photo.scale(12*u.hour) - 1.0, 1e-5)
        
    def test_vpd_low(self) : 
        """a value off the low end of the scale"""
        self.assertLess(self.xf_vpd.scale(200*u.Pa) - 0, 1e-5)
        
    def test_vpd_middle(self) :
        """a value about in the middle"""
        midrange =  (gsi.VPD_MIN + gsi.VPD_MAX)/2
        self.assertLess(self.xf_vpd.scale(midrange)- 0.5, 1e-5)
        
    def test_vpd_high(self) : 
        """a value off the top end of the scale"""
        self.assertLess(self.xf_vpd.scale(10000*u.Pa) - 1., 1e-5)
        
    def test_tmin_low(self) : 
        """a value off the low end of the scale"""
        self.assertLess(self.xf_tmin.scale(-20.*u.deg_C) - 0, 1e-5)
        
    def test_tmin_middle(self) : 
        """a value about in the middle"""
        midrange = (gsi.TMIN_MIN + gsi.TMIN_MAX) / 2.
        self.assertLess(self.xf_tmin.scale(midrange) - 0.5, 1e-5)
        
    def test_tmin_high(self) :
        """a value off the top end of the scale"""
        self.assertLess(self.xf_tmin.scale(40.*u.deg_C) - 1., 1e-5)
        
    def test_gsi_sglval(self):
        """test entire gsi calculation with scalar inputs"""
        tmin = (gsi.TMIN_MIN + gsi.TMIN_MAX)/2.
        photo = (gsi.PHOTO_MIN + gsi.PHOTO_MAX) / 2.
        vpd = (gsi.VPD_MIN + gsi.VPD_MAX) / 2.
        
        self.assertLess(self.gsi.evaluate(tmin, vpd, photo) - (0.5**3), 1e-5)
        
    def test_gsi_array(self): 
        """test entire gsi calculation with array inputs"""
        frac = np.arange(0.0, 1.0, 0.2)
        tmin = frac * (gsi.TMIN_MAX-gsi.TMIN_MIN) + gsi.TMIN_MIN
        photo = frac * (gsi.PHOTO_MAX-gsi.PHOTO_MIN) + gsi.PHOTO_MIN
        vpd = (1-frac) * (gsi.VPD_MAX-gsi.VPD_MIN) + gsi.VPD_MIN
        
        i_gsi = self.gsi.evaluate(tmin,vpd, photo)
        
        for i in range(len(frac)) :
            self.assertLess(i_gsi[i] - (frac[i]**3), 1e-5)