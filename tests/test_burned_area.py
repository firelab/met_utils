import unittest
import burned_area as ba
import numpy as np

class TestLandcover ( unittest.TestCase)  :
    def test_basic(self) : 
        """ not too hard """
        v = [ 1, 4, 6,7, 11, 16 ] 
        expected = [0, 2, 4, 6]
        result = ba.landcover_classification(v)
        self.assertTrue(np.all( expected==result))

    def test_no_forest(self) : 
        """omit forested landcover types"""
        v = [  6,7, 11, 16 ] 
        expected = [0,0,2,4]
        result = ba.landcover_classification(v)
        self.assertTrue(np.all( expected==result))

    def test_no_nonforest(self) : 
        """omit nonforested landcover types"""
        v = [ 1, 4, 11, 16 ] 
        expected = [0,2,2,4]
        result = ba.landcover_classification(v)
        self.assertTrue(np.all( expected==result))
        
    def test_no_other(self) : 
        """omit "other" land classification types"""
        v = [ 1, 4, 6,7 ] 
        expected = [0,2,4,4]
        result = ba.landcover_classification(v)
        self.assertTrue(np.all( expected==result))
       
