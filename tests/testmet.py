import unittest
import met_functions as met
import numpy as np

class TestMetFunctions (unittest.TestCase) :
    
    def setUp(self) : 
        """Tabulated values of saturation vapor pressure taken
        from the CRC Handbook of Chemistry and physics, 77th edition 1996-1997, 
        pg 6-13"""
        self.temp_c = np.array([0,5,10,15,20,25,30,35,40,45,50])
        self.vp_kpa   = np.array([0.61129, 0.87260, 1.2281, 1.7056, 2.3388,
                          3.1690, 4.2455, 5.6267, 7.3814, 9.5898, 12.344])
        
    
    def test_calc_vp(self) :
        """Ensure that the approximation formulas produce data which matches 
        the tabulated values of the vapor pressure of water. Values taken
        from the CRC Handbook of Chemistry and physics, 77th edition 1996-1997, 
        pg 6-13"""
        test_vp_kpa = met.calc_vp(self.temp_c) / 1000.0

        for i in range(len(test_vp_kpa)): 
            self.assertAlmostEqual(self.vp_kpa[i], test_vp_kpa[i], places=1,
               msg= "Failure on calc_vp(%f) == %f (got %f)" % (self.temp_c[i],self.vp_kpa[i],test_vp_kpa[i]))
               
    def test_calc_tdew(self):
        
               
suite = unittest.TestLoader().loadTestsFromTestCase(TestMetFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)