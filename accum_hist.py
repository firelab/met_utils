import numpy as np
import numpy.ma as ma
import qty_index as q


class AccumulatingHistogramdd (object)  :
    """computes an n-dimensional histogram, one observation at a time
    
    An accumulating histogram is a histogram which does not have access to the
    entire dataset at one time. The histogram is built up by repeated calls
    to "put_record". 
    
    Because the accumulating histogram does not have access to the entire 
    dataset, it cannot calculate the min/max of each of the parameters in 
    order to select a logical binsize along each axis. The user must specify
    this when the object is created. 
    
    The count of records added to the histogram is available as the count 
    parameter. The "H" parameter is the current state of the histogram.
    """
    def __init__(self, minmax=None, dtype=np.int64) : 
        """specifies the n-dimensional structure of the histogram

        If minmax is specified, it must be an iterable where 
        each element is a tuple of (min, max, num_bins) describing a 
        single axis.
        """
        self.count = 0 
        self.total = 0
        
        if minmax is not None : 
            self._init_minmax(minmax, dtype)
            
            
    def _init_minmax(self, minmax, dtype) : 
        indexers = []
        shape = []
        for cur_minmax in minmax : 
            indexers.append(q.LinearSamplingFunction(minmaxbin=cur_minmax,includemax=True))
            shape.append(cur_minmax[2])
        self._index = q.OrthoIndexer(indexers)
        self.H = np.zeros( shape, dtype=dtype) 
                
    def put_record(self, record, weight=1) : 
        """adds a single record to the histogram with the specified weight."""
        self.H[self._index.get_index(record)] += weight
        self.count += 1
        self.total += weight
        
        
        
    
