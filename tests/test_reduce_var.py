import unittest
import numpy as np
import numpy.ma as ma
import reduce_var as rv

class TestReduceVar (unittest.TestCase) : 
    def setUp(self) : 
        self.x = np.arange(24).reshape( (8,3) )
        self.y = ma.array(self.x + 100)
        self.y[3,:] = ma.masked
        self.reduce = rv.ReduceVar(self.x.shape, 0, 4)
        
    def test_mean(self) : 
        """ensure mean value reduction works"""
        self.assertTrue( np.all(self.reduce.mean(0, self.x) ==
                            np.mean(self.x[0:4,:], axis=0)))
                            
        self.assertTrue( np.all(self.reduce.mean(1, self.x) ==
                            np.mean(self.x[4:8,:], axis=0)))
        
        self.assertTrue( np.all(self.reduce.mean(0, self.y) ==
                            np.mean(self.y[0:3,:], axis=0)))
                            
        self.assertTrue( np.all(self.reduce.mean(1, self.y) ==
                            np.mean(self.y[4:8,:], axis=0)))
                            
    def test_max(self) : 
        """ensure max value reduction works"""
        self.assertTrue( np.all(self.reduce.max(0, self.x) ==
                            np.max(self.x[0:4,:], axis=0)))
                            
        self.assertTrue( np.all(self.reduce.max(1, self.x) ==
                            np.max(self.x[4:8,:], axis=0)))
        
        self.assertTrue( np.all(self.reduce.max(0, self.y) ==
                            np.max(self.y[0:3,:], axis=0)))
                            
        self.assertTrue( np.all(self.reduce.max(1, self.y) ==
                            np.max(self.y[4:8,:], axis=0)))
                            
    def test_min(self) : 
        """ensure min value reduction works"""
        self.assertTrue( np.all(self.reduce.min(0, self.x) ==
                            np.min(self.x[0:4,:], axis=0)))
                            
        self.assertTrue( np.all(self.reduce.min(1, self.x) ==
                            np.min(self.x[4:8,:], axis=0)))
        
        self.assertTrue( np.all(self.reduce.min(0, self.y) ==
                            np.min(self.y[0:3,:], axis=0)))
                            
        self.assertTrue( np.all(self.reduce.min(1, self.y) ==
                            np.min(self.y[4:8,:], axis=0)))
                            
    def test_sum(self) : 
        """ensure sum value reduction works"""
        self.assertTrue( np.all(self.reduce.sum(0, self.x) ==
                            np.sum(self.x[0:4,:], axis=0)))
                            
        self.assertTrue( np.all(self.reduce.sum(1, self.x) ==
                            np.sum(self.x[4:8,:], axis=0)))
        
        self.assertTrue( np.all(self.reduce.sum(0, self.y) ==
                            np.sum(self.y[0:3,:], axis=0)))
                            
        self.assertTrue( np.all(self.reduce.sum(1, self.y) ==
                            np.sum(self.y[4:8,:], axis=0)))

    def test_last_value(self) : 
        """ensure last value reduction works"""
        self.assertTrue( np.all(self.reduce.last_val(0, self.x) ==
                            self.x[3,:]))
                            
        self.assertTrue( np.all(self.reduce.last_val(1, self.x) ==
                            self.x[7,:]))
        
        self.assertTrue( np.all(self.reduce.last_val(0, self.y).mask))
                            
        self.assertTrue( np.all( self.reduce.last_val(1, self.y) ==
                            self.y[7,:]))
                            
    def test_size_calc(self) : 
        """ensure we calculate the reduced axis size correctly"""
        self.assertEqual(self.reduce.reduced, 2)