import unittest
import numpy as np
import accum_hist as ah


class TestAccumulatingHistogram (unittest.TestCase) : 
    def setUp(self) : 
        self.x = np.array( [ [0,0,0],
                             [0,0,1],
                             [0,0,2],
                             [0,0,0],
                             [1,0,2],
                             [1,1,0],
                             [5,2,0] ] )
        self.bins = [ (0,5,5), (0,2,2), (-1,5,6) ] 
                             
    def test_count(self) : 
        """does the counter work OK?"""
        hist = ah.AccumulatingHistogramdd(minmax=self.bins)
        for i in range(self.x.shape[0]) : 
            self.assertEqual(hist.count, i)
            hist.put_record(self.x[i,:])
            
    def test_noweights_total(self) : 
        """do we accumulate a total correctly?"""
        hist = ah.AccumulatingHistogramdd(minmax=self.bins)
        for i in range(self.x.shape[0]) : 
            self.assertEqual(hist.total, i)
            hist.put_record(self.x[i,:])
            
    def test_weights_total(self) : 
        """do we accumulate a total correctly when using weights?"""
        hist = ah.AccumulatingHistogramdd(minmax=self.bins)
        t = 0 
        for i in range(self.x.shape[0]) : 
            self.assertEqual(hist.total, t)
            hist.put_record(self.x[i,:], weight=(i+1)*100)
            t += (i+1)*100
            
    def test_bincounts(self) : 
        """do we accumulate into same bins as histogramdd?"""
        hist = ah.AccumulatingHistogramdd(minmax=self.bins)
        for i in range(self.x.shape[0]) : 
            hist.put_record(self.x[i,:])
        
            
        edges = [] 
        for ax_min, ax_max, ax_bins in self.bins  :
            edges.append(np.linspace(ax_min, ax_max, num=ax_bins+1, endpoint=True))
        counts,e = np.histogramdd(self.x, bins=edges)
        self.assertTrue(np.all(counts == hist.H))

        
class TestSparseKeyedHistogram ( unittest.TestCase ) :
    def setUp(self):
        self.threshold = 10  
        self.short_data = np.arange(self.threshold-1)
        self.long_data = np.arange(self.threshold)
        self.default_minmax = (0,self.threshold,self.threshold)
        self.minmax = [ (0,5,5), (0,2,2), (-1,5,6) ] 

    def test_default(self) : 
        h = ah.SparseKeyedHistogram(minmax=self.minmax,default_minmax=self.default_minmax)
        self.assertIs(None, h.default)
        h.put_combo( (1,2,3), self.short_data)
        self.assertEqual(len(h.default_contrib),1)
        self.assertTrue( np.all(h.default_edges == np.arange(0,self.threshold+0.5,1)))
        self.assertTrue( np.all(h.default == (1,1,1,1,1,1,1,1,1,0)))
        
        h.put_combo( (0,1,2), self.short_data)
        self.assertEqual(len(h.default_contrib),2)
        self.assertTrue( np.all(h.default == (2,2,2,2,2,2,2,2,2,0)))
        
    def test_combo(self) : 
        h = ah.SparseKeyedHistogram(minmax=self.minmax,
                            default_minmax=self.default_minmax,
                            threshold=self.threshold)
        h.put_combo( (2,1,3), self.long_data)
        self.assertIs(None, h.default)
        self.assertEqual(len(h.get_combos()),1)
        self.assertEqual( (2,1,3), h.get_combos()[0])
        