import unittest
import numpy as np
import window as w
import astropy.units as u

class TestMovingWindow (unittest.TestCase) : 
    def setUp(self) : 
        self.x = np.arange(24).reshape( (8,3) )
        self.mw = w.MovingWindow(self.x.shape, 0, 4)
        self.mwt = w.MovingWindow(self.x.T.shape, 1, 4)
    
    def test_ready(self) :
        """checks that the ready flag works""" 
        self.assertFalse(self.mw.ready())
        self.assertFalse(self.mwt.ready())
        for i in range(3) : 
            self.mw.put( np.ones((3,)) )
            self.assertFalse(self.mw.ready())
            self.mwt.put( np.ones((3,)) )
            self.assertFalse(self.mwt.ready())

                        
        self.mw.put(np.ones( (3,) ))
        self.assertTrue(self.mw.ready())
        
        self.mwt.put(np.ones( (3,) ))
        self.assertTrue(self.mwt.ready())

                
    def test_mean(self) : 
        """ checks that mean works"""
        for i in range(4) : 
            self.mw.put( self.x[i,:] ) 
            self.mwt.put( self.x.T[:,i] ) 
            
        self.assertTrue(self.mw.ready())
        self.assertTrue(self.mwt.ready())
        
        self.assertTrue(np.all(self.mw.mean() == np.mean(self.x[:4,:], axis=0)))
        self.assertTrue(np.all(self.mwt.mean() == np.mean(self.x.T[:,:4], axis=1)))
        
        self.mw.put( self.x[4,:] ) 
        self.assertTrue(self.mw.ready())
        self.assertTrue(np.all(self.mw.mean() == np.mean(self.x[1:5,:], axis=0)))

        self.mwt.put( self.x.T[:,4] ) 
        self.assertTrue(self.mwt.ready())
        self.assertTrue(np.all(self.mwt.mean() == np.mean(self.x.T[:,1:5], axis=1)))
        
                        
    def test_sum(self) : 
        """ checks that sum works"""
        for i in range(4) : 
            self.mw.put( self.x[i,:] ) 
            self.mwt.put( self.x.T[:,i] ) 
            
        self.assertTrue(self.mw.ready())
        self.assertTrue(self.mwt.ready())
        
        self.assertTrue(np.all(self.mw.sum() == np.sum(self.x[:4,:], axis=0)))
        self.assertTrue(np.all(self.mwt.sum() == np.sum(self.x.T[:,:4], axis=1)))
        
        self.mw.put( self.x[4,:] ) 
        self.assertTrue(self.mw.ready())
        self.assertTrue(np.all(self.mw.sum() == np.sum(self.x[1:5,:], axis=0)))

        self.mwt.put( self.x.T[:,4] ) 
        self.assertTrue(self.mwt.ready())
        self.assertTrue(np.all(self.mwt.sum() == np.sum(self.x.T[:,1:5], axis=1)))
        
    def test_get(self) : 
        """ensure that we can retrieve a slice of data"""
        for i in range(4) : 
            self.mw.put( self.x[i,:] ) 
            self.mwt.put( self.x.T[:,i] ) 

        self.assertTrue(self.mw.ready())
        self.assertTrue(self.mwt.ready())
        
        # get earliest slice
        self.assertTrue(np.all(self.mw.get(0) == self.x[0,:]))
        self.assertTrue(np.all(self.mwt.get(0) == self.x.T[:,0]))
        
        # get most current slice
        self.assertTrue(np.all(self.mw.get(3) == self.x[3,:]))
        self.assertTrue(np.all(self.mwt.get(3) == self.x.T[:,3]))
        
    def test_initial_value(self) : 
        """make sure we can initialize to known values"""
        mw = w.MovingWindow(self.x.shape, 0, 4, initial_value=1e20)
        self.assertTrue(np.all(mw.buffer[:] == 1e20))
        
        
    def test_unit(self) : 
        """check that our window behaves correctly for specified units"""
        self.mw.unit = u.pct
        
        # putting plain data should assume units are correct
        for i in range(4) : 
            self.mw.put( self.x[i,:] ) 
            
        # and getting should return the correct quantity object
        myslice = self.mw.get(0)
        self.assertTrue(hasattr(myslice, 'unit'))
        self.assertEqual(myslice.unit, u.pct)
        self.assertTrue(np.all(myslice[:] == self.x[0,:]*u.pct))
        
        # so should statistical functions
        self.assertTrue(np.all(self.mw.mean() == np.mean(self.x[:4,:]*u.pct, axis=0)))
        self.assertTrue(np.all(self.mw.sum() == np.sum(self.x[:4,:]*u.pct, axis=0)))
        
    def test_unit_init_val_conflict(self) : 
        """initial value should be converted to window units if specified"""
        
        # initial value is converted to window units.
        mw = w.MovingWindow(self.x.shape, 0, 4, initial_value=20*u.pct,
                    unit=u.dimensionless_unscaled)
        self.assertTrue(np.all(mw.buffer[:] == 0.2))
        
        # not specifying units on initial value involves assuming
        # units are correct
        mw = w.MovingWindow(self.x.shape, 0, 4, initial_value=20, 
                    unit=u.dimensionless_unscaled)
        self.assertTrue(np.all(mw.buffer[:] == 20))
        
        #specifying units on initial value discards units
        mw = w.MovingWindow(self.x.shape, 0, 4, initial_value=20*u.pct)
        self.assertTrue(np.all(mw.buffer[:] == 20))
        
class TestSequenceWindow(unittest.TestCase) : 
    def setUp(self) : 
        self.x = np.array( [ [ True, False, False, True ] ,
                             [ True, False, True,  False] , 
                             [ True, True,  True,  True ] ,
                             [ True, True,  True,  False] ])
                             
    def test_false_default(self) : 
        """by default, all items in buffer are false"""
        sw = w.SequenceWindow(self.x.shape, 0, 2)
        self.assertFalse(np.any(sw.buffer))
        
    def test_init_true(self) : 
        """can override so that buffer starts true"""
        sw = w.SequenceWindow(self.x.shape, 0, 2, initial_value=True)
        self.assertTrue(np.all(sw.buffer)) 
        
    def test_last_run(self) : 
        """Ensure that we can count True values from the most recent addition"""
        sw = w.SequenceWindow(self.x.shape, 0, 2)
        
        # no runs on initial object
        self.assertTrue(np.all(sw.last_run_length() == 0))
        
        sw.put(self.x[0,:])
        self.assertTrue(np.all(sw.last_run_length() == np.array([1,0,0,1])))
        
        sw.put(self.x[1,:])
        self.assertTrue(np.all(sw.last_run_length() == np.array([2,0,1,0])))
                    
        sw.put(self.x[2,:])
        self.assertTrue(np.all(sw.last_run_length() == np.array([2,1,2,1])))
        
        sw.put(self.x[3,:])
        self.assertTrue(np.all(sw.last_run_length() == np.array([2,2,2,0])))

    def test_all_runs(self) : 
        """ensure that weighted run counter works"""
        sw = w.SequenceWindow(self.x.shape, 0, 3)
        
        # no runs on initial object
        self.assertTrue(np.all(sw.all_runs() == 0))
        
        sw.put(self.x[0,:])
        self.assertTrue(np.all(sw.all_runs() == np.array([1,0,0,1])))
        
        sw.put(self.x[1,:])
        self.assertTrue(np.all(sw.all_runs() == np.array([3,0,1,1])))

        sw.put(self.x[2,:])
        self.assertTrue(np.all(sw.all_runs() == np.array([6,1,3,2])))

        sw.put(self.x[3,:])
        self.assertTrue(np.all(sw.all_runs() == np.array([6,3,6,1])))
