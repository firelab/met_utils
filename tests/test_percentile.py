import unittest
import percentile as p
import numpy as np

class TestPercentileFactory (unittest.TestCase) : 
    def setUp(self) :
        self.data = np.arange(0,201)
        self.factory = p.PercentileFactory(len(self.data))
        
    def test_single_add(self) : 
        """add all the data in one go"""
        self.factory.add_data(self.data)
        fn = self.factory.compute_percentile()
        self.assertTrue(np.all((fn.cutpoints - np.arange(0,201,2)) < 1e-6))
        
    def test_multi_add(self) : 
        """add the data in 2 or three chunks"""
        chunksize = len(self.data)
        self.factory.add_data(self.data[:chunksize])
        self.factory.add_data(self.data[chunksize:])
        fn = self.factory.compute_percentile()
        self.assertTrue(np.all((fn.cutpoints - np.arange(0,201,2)) < 1e-6))
        
    def test_reverse_add(self) : 
        """add the data in reverse order"""
        revdata = self.data[-1:0:-1]
        self.factory.add_data(revdata)
        fn = self.factory.compute_percentile()
        self.assertTrue(np.all((fn.cutpoints - np.arange(0,201,2)) < 1e-6))

    def test_edge_percentiles(self) : 
        """what are percentiles 0 and 100?"""
        self.factory.add_data(self.data)
        fn = self.factory.compute_percentile()
        self.assertEqual(fn.cutpoints[0], 0)
        self.assertEqual(fn.cutpoints[100], 200)
                