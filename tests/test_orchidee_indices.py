import unittest
import netCDF4 as nc
import orchidee_indices as oi
import astropy.units as u
import numpy as np
import os


class TestForcingDataset (unittest.TestCase) : 
    def setUp(self) : 
        outfilename = "test.nc"
        self.fd = oi.ForcingDataset('dummy.nc','test.nc')
        
        # make an in-memory netCDF dataset for testing purposes.
        fake_ds = nc.Dataset('fake', mode='w', diskless=True)
        
        tstep = fake_ds.createDimension('tstep', 16)
        x =     fake_ds.createDimension('x', 720)
        y =     fake_ds.createDimension('y', 360)
        land =  fake_ds.createDimension('land', 10)

        time =  fake_ds.createVariable('time', np.float32, dimensions=('tstep',))
        time[:] = np.arange(16.)*21600
        
        land = fake_ds.createVariable('land', np.int, dimensions=('land',))
        land[:] = [19584, 23878, 34692, 40809, 53676, 
                  60260, 98848, 119744, 160446, 170550]
        land.compress = 'y x'

        # fake latitude array
        nav_lat = fake_ds.createVariable('nav_lat', np.float32, dimensions=('y','x'))
        lat_template = np.arange(89.75,-90,-0.5)
        for i in range(720): 
            nav_lat[:,i] = lat_template
            
            
        # fake longitude array
        nav_lon = fake_ds.createVariable('nav_lon', np.float32, dimensions=('y','x'))
        lon_template = np.arange(-179.75,180, 0.5)
        for i in range(360):
            nav_lon[i,:] = lon_template
            
        
        self.fd._nc_forcing = fake_ds
        
        
    def tearDown(self) : 
        self.fd.close()
        os.remove('test.nc')
        
    def test_get_forcing(self): 
        """do we get the thing we're initialized with?"""
        self.assertEqual(len(self.fd.get_forcing().dimensions["tstep"]), 16)
        
    def test_get_axes(self) : 
        """we convert to the correct cells?"""
        ca = self.fd.get_land_axes()
        self.assertTrue(np.all(ca.getIndices(0)   == (0,0) ))
        self.assertTrue(np.all(ca.getIndices(720) == (1,0) ))

    def test_get_indices(self): 
        """checks that the indices are presented in the expected format"""
        xy = self.fd.get_xy_indices()
        test_grid = np.zeros((360,720))
        test_grid[xy] = 5
        
        # in all, ten cells should have been set to 5
        self.assertEqual(test_grid.sum(), 5*10)
        
        # land[0] (19584) == (27,144)
        self.assertEqual(test_grid[27,144], 5)
        
    def test_indices_ncdf(self) : 
        """check that we can use indices on netcdf variables"""
        
        forcing = self.fd.get_forcing()
        nav_lon = forcing.variables['nav_lon']
        xy = self.fd.get_xy_indices()
        
        test_lon = nav_lon[xy]
        self.assertEqual(len(test_lon), 10)
        
        self.assertEqual(nav_lon[27,144], test_lon[0])
        
    def test_longitude(self) : 
        """check that longitudes are retrieved correctly and have units"""
        lons = self.fd.get_longitudes()
        
        forcing = self.fd.get_forcing()
        nav_lon = forcing.variables['nav_lon']
        xy = self.fd.get_xy_indices()
        test_lons = nav_lon[xy] * u.deg
        
        self.assertTrue(np.all(test_lons == lons))
        
    def test_get_timestep(self) : 
        """check that we calculate the timestep correctly"""
        timestep = self.fd.get_timestep()
        self.assertEqual(timestep, 21600*u.s)
        