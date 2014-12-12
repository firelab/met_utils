"""Code to aggregate values along one dimension of an ndarray."""
import numpy as np
import numpy.ma as ma


class ReduceVar (object) : 
    """Aggregates along a user specified axis in a user-specified manner.
    
    This defines a controller object which facilitates consistent data 
    reduction across multiple variables. It remembers key attributes like
    the shape of the variable, the axis to be aggregated, and the reduction
    factor. It then computes the appropriate descriptive statistics to arrive 
    at the value which is representative of the interval requested. 
    """
    
    def __init__(self, shape, agg_axis, reduction) : 
        """specifies the aggregation characteristics
        
        User specifies the shape of variables which this object can work on, 
        the agg_axis which is to be reduced, and the reduction factor.
        """
        self.shape     = shape
        self.agg_axis  = agg_axis
        self.reduction = reduction
        self.reduced   = shape[agg_axis] / reduction
        
    def _selection(self, i_agg) :
        """computes and returns an index to select the region to be summarized
        
        caller specifies the index along the reduced aggregation axis, and 
        the slice expression into the original variable is returned.
        """
        i = [ slice(None,None,None)] * len(self.shape) 
        begin = i_agg * self.reduction
        end = begin + self.reduction
        i[self.agg_axis] = slice(begin, end)
        return i
        
    def mean(self, i_agg, v) : 
        """computes the i_agg-th slice of reduced data from v using the mean"""
        i = self._selection(i_agg)
        return ma.mean(v[i], axis=self.agg_axis)
        
    def max(self, i_agg, v) : 
        """computes the i_agg-th slice of reduced data from v using the max"""
        i = self._selection(i_agg)
        return ma.max(v[i], axis=self.agg_axis)
        
    def min(self, i_agg, v) : 
        """computes the i_agg-th slice of reduced data from v using the min"""
        i = self._selection(i_agg)
        return ma.min(v[i], axis=self.agg_axis)
        
    def sum(self, i_agg, v) : 
        """computes the i_agg-th slice of reduced data from v using the sum"""
        i = self._selection(i_agg)
        return ma.sum(v[i], axis=self.agg_axis)

    def last_val(self, i_agg, v) : 
        """computes the i_agg-th slice of reduced data from v using the last value"""
        i = self._selection(i_agg)
        i[self.agg_axis] = (i[self.agg_axis].stop-1)
        return v[i]


class CutpointReduceVar ( ReduceVar )  :
    """Allows a user to specify an array of cutpoints which define bins"""
    
    def __init__(self, shape, agg_axis, cutpoints)  :
        """Creates a new object to segment an array using a series of cutpoints
        
        Cutpoints are the edges, and the min/max must be included. There are 
        always one fewer intervals than cutpoints. The i_agg index refers to
        an interval defined by two bounding cutpoints (e.g., i_agg=0 means to 
        select all data points between cutpoints 0 and 1.)
        """
        self.shape = shape
        self.agg_axis = agg_axis
        self.cutpoints = cutpoints
        self.reduced = len(cutpoints) - 1

        
    def _selection(self, i_agg) :
        """computes and returns an index to select the region to be summarized
        
        caller specifies the index along the reduced aggregation axis, and 
        the slice expression into the original variable is returned.
        
        i_agg of 0 means to select all values between cutpoints[0] and cutpoints[1]
        """
        i = [ slice(None,None,None)] * len(self.shape) 
        begin = self.cutpoints[i_agg]
        end = self.cutpoints[i_agg+1]
        i[self.agg_axis] = slice(begin, end)
        return i
        