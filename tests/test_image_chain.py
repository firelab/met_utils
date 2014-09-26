import unittest
import numpy as np 
import image_chain as ic

class TestNdarrayRoundRobin(unittest.TestCase) : 
    def setUp(self) : 
        self.ndarray = np.array([ [1,2], [3,4], [5,6] ])
        self.rr = self.ndarray[:].view(ic.NdarrayRoundRobin)
        
    def test_shape(self) : 
        """shape is same as underlying array"""
        self.assertEqual(len(self.rr.shape), 2, "Dimensions==2")
        self.assertEqual(self.rr.histdim, 1, "histdim is last")
        self.assertEqual(self.rr.shape[self.rr.histdim], 2, "shape is (3,2)")
        
    def test_nochange(self): 
        self.assertTrue(np.array_equal(self.rr,self.ndarray))
        
    def test_get(self) : 
        """make sure getitem operator works"""
        print self.rr[0,1]
        self.assertEqual(self.rr[0,1], 2, "got %s" %str(self.rr[0,1]))

    def test_set(self) : 
        """make sure setitem operator works"""
        self.rr[0,1] = 55
        self.assertEqual(self.rr[0,1], 55)
        self.assertEqual(self.ndarray[0,1], 55)
        
    def test_next(self) :
        """Advance the zero point and check indexing"""
        self.rr.next()
        self.assertEqual(self.rr[0,1], 1)
        self.assertFalse(self.rr[0,1] == self.ndarray[0,1])
        self.rr[0,1] = 55
        self.assertEqual(self.rr[0,1], 55)
        self.assertEqual(self.ndarray[0, 0], 55)
        
    def test_buffer_same(self): 
        """Advancing zero point doesn't change buffer..."""
        self.rr.next() 
        self.assertTrue(np.all(self.rr == self.ndarray))