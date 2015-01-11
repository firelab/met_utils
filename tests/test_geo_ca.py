import unittest
import geo_ca 
import netCDF4 as nc
import numpy as np
import numpy.ma as ma


Y_SIZE = 360
X_SIZE = 720
DELTA_X = 0.5
DELTA_Y = 0.5

def compress_grid_coords(x,y, mode="F") :
    """compresses using y as the slowest changing dimension"""
    retval = (y*X_SIZE)+x
    if mode == "F" : 
        retval += 1
    return retval
    
def uncompress_grid_coords(i, mode="F") : 
    if mode == "F" : 
        i -= 1
    x = i % X_SIZE
    y = floor(i / X_SIZE)
    return (x,y)
    
def compressed_ncfile(gridcoords, mode="F", c_dim='land') : 
    """creates in-memory netcdf file for testing compressed axes"""
    ncfile = nc.Dataset('temp.nc', mode='w', diskless=True)
    
    # 720x360 grid, uncompressed
    ncfile.createDimension('x', X_SIZE)
    ncfile.createDimension('y', Y_SIZE)
    
    # compress and store the provided grid coordinates
    land = [ compress_grid_coords(x,y,mode) for x,y in gridcoords ] 
    ncfile.createDimension(c_dim, len(land))
    v_land = ncfile.createVariable(c_dim, np.int32, dimensions=(c_dim,))
    v_land[:] = land
    v_land.compress = 'y x'
    
    return ncfile
    
def geo_ncfile(ncfile) : 
    """adds nav_lat and nav_lon to the netcdf testfile"""
    lats = ncfile.createVariable('nav_lat', np.float32, dimensions=('y','x'))
    lons = ncfile.createVariable('nav_lon', np.float32, dimensions=('y','x'))
    
    latvals = np.arange((-90+DELTA_Y/2), 90, DELTA_Y)
    lonvals = np.arange((-180+DELTA_X/2), 180, DELTA_X)
    
    for i in range(X_SIZE) : 
        lats[:,i] = latvals
        
    for i in range(Y_SIZE): 
        lons[i,:] = lonvals
        
    

class TestGeoCA (unittest.TestCase) : 
    def setUp(self) : 
        """construct a fake netcdf file for testing compression"""
        self.gridcoords = [ (10, 10), (20,20), (100,100), (200,200) ]
        ncfile = compressed_ncfile(self.gridcoords)
        geo_ncfile(ncfile)
        self.ncfile = ncfile
        self.grid = np.arange(0,X_SIZE*Y_SIZE).reshape((Y_SIZE,X_SIZE))
        self.vec = np.arange(0,len(self.gridcoords))
        self.gca = geo_ca.GeoCompressedAxes(self.ncfile, 'land')
        
    def tearDown(self) : 
        self.ncfile.close()
        
    def test_no_clip(self) : 
        """make sure we get all our data if no mask is set"""
        testvec = self.gca.compress(self.grid)
        self.assertEqual(ma.count(testvec), len(self.gridcoords))
        
    def test_ll_corner(self) : 
        """test that we filter out everything but ll corner"""
        self.gca.set_clip_box(-90,-82,-180,-173)
        testvec = self.gca.compress(self.grid)
        self.assertTrue(self.gca.is_masked())
        testmask = self.gca.get_vec_mask()
        self.assertEqual(np.count_nonzero(testmask), 3)
        self.assertEqual(ma.count(testvec), 1)
        
    def test_remove_clip_box(self) : 
        """test that we can remove the clip box once set."""
        self.gca.set_clip_box(-90,-75,-180,-165)
        testvec = self.gca.compress(self.grid)
        self.assertTrue(self.gca.is_masked())
        self.assertEqual(ma.count(testvec), 2)
        
        self.gca.remove_mask()
        self.assertFalse(self.gca.is_masked())
        testvec = self.gca.compress(self.grid)
        self.assertEqual(ma.count(testvec), 4)
        
    def test_reset_clip_box(self) : 
        """test that we can define a different clip box once set"""
        self.gca.set_clip_box(-90,-82,-180,-173)
        testvec = self.gca.compress(self.grid)
        self.assertTrue(self.gca.is_masked())
        testmask = self.gca.get_vec_mask()
        self.assertEqual(np.count_nonzero(testmask), 3)
        self.assertEqual(ma.count(testvec), 1)

        self.gca.set_clip_box(-90,-75,-180,-165)
        testvec = self.gca.compress(self.grid)
        self.assertTrue(self.gca.is_masked())
        self.assertEqual(ma.count(testvec), 2)
        testmask = self.gca.get_vec_mask()
        self.assertEqual(np.count_nonzero(testmask), 2)
        self.assertEqual(ma.count(testvec), 2)

        
        
        
        