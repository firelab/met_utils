import unittest
from astropy import units as u
from astropy.units import imperial as iu
import fuelmoisture as fm
import qty_index as qi
import numpy as np
import numpy.ma as ma

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
                               
class TestDiurnalLocalTimeStatistics(unittest.TestCase) : 
    def setUp(self) : 
        # six points, four samples/day, 4 days
        self.test_data = np.arange(96).reshape( (6,16) )
        self.time_axis = 1
        self.timestep = 6*u.hour
        self.lons = np.arange(6)*45*u.deg - 135*u.deg
        
    def test_creation(self) : 
        """test creation and initialization of object"""
        x = fm.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
                                          
    def test_attributes(self) : 
        """test initialization of various attributes"""
        x = fm.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        self.assertEqual(x.diurnal_window, 4)
        self.assertEqual(x.time_axis, self.time_axis)
        self.assertEqual(x.timestep, self.timestep)
        self.assertTrue(np.all(self.lons == x.lons))
        self.assertTrue(np.all(x.buffer.shape == np.array((6,8))))

    def test_mask(self) :
        """test initialization of mask"""
        x = fm.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        lsf = qi.LongitudeSamplingFunction(24*u.hour/self.timestep, 13*u.hour)
        i_lons = lsf.get_index(self.lons)
        
        afternoon = i_lons[0]
        self.assertTrue(afternoon == 3 )
        mask_test = [ not((i>afternoon) and (i <= afternoon+4)) for i in range(8) ]
        self.assertTrue(np.all(x.mask[0,:] == mask_test))
        
    def test_mean(self) :
        """test the masked mean function"""
        x = fm.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        first_point = ma.array(self.test_data[0,:8], mask=x.mask[0,:]).mean()
        self.assertEqual(first_point, x.mean()[0])
        first_point = self.test_data[0,4:8].mean()
        self.assertEqual(first_point, x.mean()[0])

    def test_min(self) :
        """test the masked min function"""
        x = fm.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        first_point = ma.array(self.test_data[0,:8], mask=x.mask[0,:]).min()
        self.assertEqual(first_point, x.min()[0])
        first_point = self.test_data[0,4:8].min()
        self.assertEqual(first_point, x.min()[0])
 
    def test_max(self) :
        """test the masked max function"""
        x = fm.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        first_point = ma.array(self.test_data[0,:8], mask=x.mask[0,:]).max()
        self.assertEqual(first_point, x.max()[0])
        first_point = self.test_data[0,4:8].max()
        self.assertEqual(first_point, x.max()[0])
