import unittest
import astropy.units as u
import astropy.coordinates as c
import qty_index as q
import numpy as np
import numpy.ma as ma
import netCDF4 as nc


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

class TestLookupTable (unittest.TestCase) : 
    def test_longitudes(self) : 
        """exercises the case where we are a longitude lookup"""
        
        lut = q.LookupTable(-179.75*u.deg, 0.5*u.deg, np.arange(720))
        self.assertEqual(lut.get( 0*u.deg ), 359)             # middle bin
        self.assertEqual(lut.get( -179.75 * u.deg) , 0) # first bin
        self.assertEqual(lut.get( 179.75 * u.deg), 719) # last bin
        

class TestDiurnalLocalTimeStatistics(unittest.TestCase) : 
    def setUp(self) : 
        # six points, four samples/day, 4 days
        self.test_data = np.arange(96).reshape( (6,16) )
        self.time_axis = 1
        self.timestep = 6*u.hour
        self.lons = np.arange(6)*45*u.deg - 135*u.deg
        
    def test_creation(self) : 
        """test creation and initialization of object"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
                                          
    def test_attributes(self) : 
        """test initialization of various attributes"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        self.assertEqual(x.diurnal_window, 4)
        self.assertEqual(x.time_axis, self.time_axis)
        self.assertEqual(x.timestep, self.timestep)
        self.assertTrue(np.all(self.lons == x.lons))
        self.assertTrue(np.all(x.buffer.shape == np.array((6,8))))

    def test_mask(self) :
        """test initialization of mask"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        lsf = q.LongitudeSamplingFunction(24*u.hour/self.timestep, 13*u.hour)
        i_lons = lsf.get_index(self.lons)
        
        afternoon = i_lons[0]
        self.assertTrue(afternoon == 3 )
        mask_test = [ not((i>afternoon) and (i <= afternoon+4)) for i in range(8) ]
        self.assertTrue(np.all(x.mask[0,:] == mask_test))
        
    def test_mean(self) :
        """test the masked mean function"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        first_point = ma.array(self.test_data[0,:8], mask=x.mask[0,:]).mean()
        self.assertEqual(first_point, x.mean()[0])
        first_point = self.test_data[0,4:8].mean()
        self.assertEqual(first_point, x.mean()[0])

    def test_min(self) :
        """test the masked min function"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        first_point = ma.array(self.test_data[0,:8], mask=x.mask[0,:]).min()
        self.assertEqual(first_point, x.min()[0])
        first_point = self.test_data[0,4:8].min()
        self.assertEqual(first_point, x.min()[0])
 
    def test_max(self) :
        """test the masked max function"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        first_point = ma.array(self.test_data[0,:8], mask=x.mask[0,:]).max()
        self.assertEqual(first_point, x.max()[0])
        first_point = self.test_data[0,4:8].max()
        self.assertEqual(first_point, x.max()[0])
        
    def test_next(self) : 
        """test that next() advances to the next day"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        x.next()
        self.assertEqual(x.cur_day, 3)
        self.assertTrue(np.all(x.buffer[0,:]==self.test_data[0,4:12]))

    def test_nonsequential_load(self) : 
        """checks that loading days nonsequentially works OK"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons, sequential=False)
        self.assertEqual(None, x.cur_day)
        self.assertEqual(None, x.buffer)
        x.load_day(3)
        self.assertTrue(np.all(x.buffer[0,:]==self.test_data[0,8:16]))
        x.load_day(1)
        self.assertTrue(np.all(x.buffer[0,:]==self.test_data[0,:8]))
        
    def test_nonsequential_stats(self) : 
        """checks that we can compute statistics after loading nonsequentially"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons, sequential=False)
        x.load_day(3)
        
        # mean
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).mean()
        self.assertEqual(first_point, x.mean()[0])
        first_point = self.test_data[0,12:16].mean()
        self.assertEqual(first_point, x.mean()[0])
        
        # max
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).max()
        self.assertEqual(first_point, x.max()[0])
        first_point = self.test_data[0,12:16].max()
        self.assertEqual(first_point, x.max()[0])
        
        # min
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).min()
        self.assertEqual(first_point, x.min()[0])
        first_point = self.test_data[0,12:16].min()
        self.assertEqual(first_point, x.min()[0])
        
    def test_ref_val(self) : 
        """checks that we pick the correct reference value for all lon points."""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        lsf = q.LongitudeSamplingFunction(24*u.hour/self.timestep, 13*u.hour)
        i_lons = lsf.get_index(self.lons) + 4
        
        result = np.empty( (len(i_lons),) )
        for i in range(len(i_lons)) : 
            result[i] = self.test_data[i, i_lons[i]]
        
        self.assertTrue(np.all(result==x.ref_val()))
        
        # now switch test data so that there's more land points than time points
        # and the axes are reversed
        self.test_data = np.arange(100*16).reshape(16,100)
        self.time_axis = 0
        self.lons = np.arange(100)*2*u.deg - 135*u.deg
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        lsf = q.LongitudeSamplingFunction(24*u.hour/self.timestep, 13*u.hour)
        i_lons = lsf.get_index(self.lons) + 4
        
        result = np.empty( (len(i_lons),) )
        for i in range(len(i_lons)) : 
            result[i] = self.test_data[i_lons[i],i]
        
        self.assertTrue(np.all(result==x.ref_val()))
 
        
        
    def test_preceeding_day(self): 
        """tests to ensure the correct data is returned"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        lsf = q.LongitudeSamplingFunction(24*u.hour/self.timestep, 13*u.hour)
        i_lons = lsf.get_index(self.lons) 
        
        result = np.empty( (len(i_lons),4) )
        for i in range(len(i_lons)) : 
            result[i,:] = self.test_data[i, i_lons[i]+1:i_lons[i]+5]
        
        self.assertTrue(np.all(result==x.get_preceeding_day()))
        
    def test_netcdf_source(self) : 
        """test to ensure we can use netcdf variable as source"""
        d = nc.Dataset('test.nc', 'w', diskless=True)
        
        d.createDimension('time',16)
        d.createDimension('land',6)
        
        v = d.createVariable('test', self.test_data.dtype, dimensions=('land','time'))
        v[:] = self.test_data
        
        
        x = q.DiurnalLocalTimeStatistics(v, self.time_axis, 
                                          self.timestep, self.lons)

        x.load_day(3)
        
        # mean
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).mean()
        self.assertEqual(first_point, x.mean()[0])
        first_point = self.test_data[0,12:16].mean()
        self.assertEqual(first_point, x.mean()[0])
        
        # max
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).max()
        self.assertEqual(first_point, x.max()[0])
        first_point = self.test_data[0,12:16].max()
        self.assertEqual(first_point, x.max()[0])
        
        # min
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).min()
        self.assertEqual(first_point, x.min()[0])
        first_point = self.test_data[0,12:16].min()
        self.assertEqual(first_point, x.min()[0])
        
    def test_template_create(self) :
        """make sure we can create new instances from a template."""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        
        v = np.ones( self.test_data.shape)                                                                    
        y = q.DiurnalLocalTimeStatistics(v, template=x) 
        self.assertEqual(self.time_axis, y.time_axis)
        self.assertEqual(self.timestep,  y.timestep)
        self.assertTrue(np.all(self.lons == y.lons))
        self.assertTrue(np.all(x.i_ref == y.i_ref))
        self.assertTrue(np.all(x.mask == y.mask))
        self.assertTrue(np.all(x.source == self.test_data))
        self.assertTrue(np.all(y.source == v))
        
    def test_unit(self) : 
        """make sure that returned data have correct units attached"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons, unit=u.Pa)
                                          
        self.assertEqual(x.get_utc_day().unit, u.Pa)
        self.assertEqual(x.get_preceeding_day().unit, u.Pa)
        self.assertEqual(x.get_buffer().unit, u.Pa)
        
        # repeat for "no units" case
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
        self.assertFalse(hasattr(x.get_utc_day(), "unit"))
        self.assertFalse(hasattr(x.get_preceeding_day(), "unit"))
        self.assertFalse(hasattr(x.get_buffer(), "unit"))
        
    def test_units_on_statistics(self) :
        """make sure statistics functions return Quantities by default"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons, sequential=False,
                                          unit=u.Pa)
        x.load_day(3)
        
        # mean
        mean = x.mean()
        self.assertTrue(hasattr(mean, "unit"))
        self.assertEqual(mean.unit, u.Pa)
        
        # max
        mx = x.max()
        self.assertTrue(hasattr(mx, "unit"))
        self.assertEqual(mx.unit, u.Pa)
        
        # min
        mn = x.min()
        self.assertTrue(hasattr(mn, "unit"))
        self.assertEqual(mn.unit, u.Pa)
        
        
    def test_forego_units(self) : 
        """make sure we can forego units if needed"""
        x = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons, unit=u.Pa)
                                          
        self.assertEqual(x.get_utc_day().unit, u.Pa)
        self.assertEqual(x.get_preceeding_day().unit, u.Pa)
        self.assertEqual(x.get_buffer().unit, u.Pa)
        x.load_day(3)
        
        # mean
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).mean()
        self.assertEqual(first_point, x.mean(unitted=False)[0])
        first_point = self.test_data[0,12:16].mean()
        self.assertEqual(first_point, x.mean(unitted=False)[0])
        
        # max
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).max()
        self.assertEqual(first_point, x.max(unitted=False)[0])
        first_point = self.test_data[0,12:16].max()
        self.assertEqual(first_point, x.max(unitted=False)[0])
        
        # min
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).min()
        self.assertEqual(first_point, x.min(unitted=False)[0])
        first_point = self.test_data[0,12:16].min()
        self.assertEqual(first_point, x.min(unitted=False)[0])
        
        
    def test_template_statistics(self) : 
        """ensure that statistics fns work on template creation"""
        v = np.ones( self.test_data.shape)                                                                    
        y = q.DiurnalLocalTimeStatistics(v, self.time_axis, 
                                          self.timestep, self.lons)
        x = q.DiurnalLocalTimeStatistics(self.test_data, template=y)
        x.load_day(3)
        
        # mean
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).mean()
        self.assertEqual(first_point, x.mean()[0])
        first_point = self.test_data[0,12:16].mean()
        self.assertEqual(first_point, x.mean()[0])
        
        # max
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).max()
        self.assertEqual(first_point, x.max()[0])
        first_point = self.test_data[0,12:16].max()
        self.assertEqual(first_point, x.max()[0])
        
        # min
        first_point = ma.array(self.test_data[0,8:16], mask=x.mask[0,:]).min()
        self.assertEqual(first_point, x.min()[0])
        first_point = self.test_data[0,12:16].min()
        self.assertEqual(first_point, x.min()[0])
        

class TestDLTSGroup (unittest.TestCase) : 
    def setUp(self) : 
        # six points, four samples/day, 4 days
        self.test_data = np.arange(96).reshape( (6,16) )
        self.time_axis = 1
        self.timestep = 6*u.hour
        self.lons = np.arange(6)*45*u.deg - 135*u.deg
        self.dlts = q.DiurnalLocalTimeStatistics(self.test_data, self.time_axis, 
                                          self.timestep, self.lons)
                                          
    def test_template_ready(self): 
        """verify that group object correctly advertises template creation readiness"""
        grp = q.DLTSGroup()
        self.assertFalse(grp.template_ready())
        grp.add("base", self.dlts)
        self.assertTrue(grp.template_ready())
        
    def test_template_creation(self): 
        """verify that group correctly creates a template"""
        grp = q.DLTSGroup()
        self.assertEqual(grp.template, None)
        
        v = np.ones( self.test_data.shape)   
        grp.add("base", self.dlts)
        self.assertEqual(grp.template, self.dlts)

        y = grp.create("bogus", v, u.Pa)
                
        self.assertEqual(self.time_axis, y.time_axis)
        self.assertEqual(self.timestep,  y.timestep)
        self.assertTrue(np.all(self.lons == y.lons))
        self.assertTrue(np.all(self.dlts.i_ref == y.i_ref))
        self.assertTrue(np.all(self.dlts.mask == y.mask))
        self.assertTrue(np.all(self.dlts.source == self.test_data))
        self.assertTrue(np.all(y.source == v))
                                                                       

        
        

    