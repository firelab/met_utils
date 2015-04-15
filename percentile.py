import numpy as np
import scipy.stats as stats
import qty_index as qi

# Class to calculate a percentile mapping function based on data.
class PercentileFactory (object) : 
    def __init__(self, npts) : 
        self.npts = npts
        self.i_pts = 0 
        self.data = np.zeros( (npts,) )
        
    def add_data(self, points) : 
        """add a series of data to the collection so far."""
        num_add = len(points) 
        self.data[self.i_pts:self.i_pts+num_add] = points
        self.i_pts += num_add
        
    def compute_percentile(self) :
        """computes and returns a percentile mapping function"""
        cutpoints = stats.scoreatpercentile(self.data, range(0,101))
        return qi.IntervalSamplingFunction(cutpoints)
        