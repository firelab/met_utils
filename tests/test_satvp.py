import unittest
import satvp
import numpy as np
from astropy import units as u

class TestSaturationVapor (unittest.TestCase) :
    
    def setUp(self) : 
        """Tabulated values of saturation vapor pressure taken
        from the CRC Handbook of Chemistry and physics, 77th edition 1996-1997, 
        pg 6-13"""
        self.temp = np.array([0,5,10,15,20,25,30,35,40,45,50]) * u.deg_C
        self.vp   = np.array([0.61129, 0.87260, 1.2281, 1.7056, 2.3388,
                          3.1690, 4.2455, 5.6267, 7.3814, 9.5898, 12.344]) * u.kPa
        
    
    def test_calc_vp(self) :
        """Ensure that the approximation formulas produce data which matches 
        the tabulated values of the vapor pressure of water. Tolerance is 
        0.1 kPa."""
        
        for method in satvp.vp_calcs.keys() :
            calc = satvp.vp_calcs[method]
            
            # only reason to explicitly convert to kPa is to ensure the precision 
            # of the comparison is expressed in kPa.
            test_vp = calc.calc_vp(self.temp).to(u.kPa)

            for i in range(len(test_vp)): 
                self.assertTrue((self.vp[i]-test_vp[i]) < 0.1*u.kPa,
                msg= "Failure on {}.calc_vp({:3.1}) == {:3.1} (got {:3.1})".format(method, self.temp[i],self.vp[i],test_vp[i]))
               
    def test_inverse(self) : 
        """Ensures that calc_vp and calc_tdew are inverse functions"""
        for method in satvp.vp_calcs.keys() : 
            calc = satvp.vp_calcs[method]
            
            self.assertTrue((20.*u.deg_C-calc.calc_tdew(calc.calc_vp(20.*u.deg_C))) < 1e-6 * u.deg_C,
                msg = "%s does not implement an inverse function pair" % method)
                
    def test_dictionary(self): 
        """Sanity check"""
        self.assertEqual(len(satvp.vp_calcs), 14)
        
    def test_kelvin(self) : 
        """check to ensure that we can calculate vp using Kelvin temps"""
        for method in satvp.vp_calcs.keys() :
            calc = satvp.vp_calcs[method]
            
            # only reason to explicitly convert to kPa is to ensure the precision 
            # of the comparison is expressed in kPa.
            test_vp = calc.calc_vp(self.temp.to(u.K, equivalencies=u.temperature())).to(u.kPa)

            for i in range(len(test_vp)): 
                self.assertTrue((self.vp[i]-test_vp[i]) < 0.1*u.kPa,
                msg= "Failure on {}.calc_vp({:3.1}) == {:3.1} (got {:3.1})".format(method, self.temp[i],self.vp[i],test_vp[i]))

class TestNegativeTempSaturationVapor (unittest.TestCase) :
    
    def setUp(self) : 
        """Tabulated values of saturation vapor pressure taken
        from [1], Table 2.1
        
        These are still saturation vapor pressure over a plane of liquid water, 
        not ice. [1] warns: "There is some uncertainty about the values of e_s
        for T<0C owing to a lack of experimental data." Physically speaking, 
        over land, it will be hard to find exposed liquid water if the temperatures 
        drop too far below zero for too long.
        
        Saturation vapor pressures over ice are smaller than saturation vapor 
        pressures over water.

        1.Rogers, R. R. A Short Course in Cloud Physics, Third Edition. (Pergamon, 1989).
        """
        self.temp = np.array([-5, -10, -15, -20, -25, -30, -35, -40]) * u.deg_C
        self.vp   = np.array([421.84, 286.57, 191.44, 125.63, 80.9, 
                              51.06, 31.54, 19.05]) * u.Pa
        
        
               
#suite = unittest.TestLoader().loadTestsFromTestCase(TestSaturationVapor)
#unittest.TextTestRunner(verbosity=2).run(suite)