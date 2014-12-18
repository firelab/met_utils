import numpy as np
import numpy.ma as ma
import qty_index as q


def _init_indexers(minmax) :
    indexers = []
    for cur_minmax in minmax : 
        indexers.append(q.LinearSamplingFunction(minmaxbin=cur_minmax,includemax=True))
    return q.OrthoIndexer(indexers)


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
        self._index = _init_indexers(minmax)
        shape = []
        for cur_minmax in minmax : 
            shape.append(cur_minmax[2])
        self.H = np.zeros( shape, dtype=dtype) 
                
    def put_record(self, record, weight=1) : 
        """adds a single record to the histogram with the specified weight."""
        self.H[self._index.get_index(record)] += weight
        self.count += 1
        self.total += weight
        
        
class SparseKeyedHistogram (object) : 
    """Stores a histogram at each bin
    
    Implements the concept where a combination of parameters selects an
    entire histogram. Unique combinations of parameters are expected to
    sparsely populate the parameter space. The data from which the histograms
    are calculated are provided all at once, but each unique combination of 
    parameters is populated atomically, by providing all of the data contributing
    to that bin simultaneously.
    
    This class also maintains the notion of a default histogram which 
    aggregates data from all cells having less than a specified threshhold of 
    observations. The default histogram is maintained as the "default" attribute
    which can be accessed directly. The individual histograms for each 
    combination of parameters are accessed using the get_histogram and get_edges
    functions.
    """        
    def __init__(self, minmax=None, threshold=50, bins=25,
                    default_minmax=None) : 
        self._index = _init_indexers(minmax)
        self.bins = bins
        self.threshold = threshold
        self.default = None
        self.default_minmax = default_minmax
        self.default_contrib = {}
        self.histograms = {}
        self._init_default(threshold)
            
    def _init_default(self, threshold) : 
        """produce indices for all combos lacking enough data"""
        binsize = float(self.default_minmax[1]-self.default_minmax[0])/self.default_minmax[2]
        self.default_edges = np.arange(self.default_minmax[0],
                                       self.default_minmax[1]+(binsize/2),
                                       binsize)
                                       
                                       
        
    def put_combo(self, combo, data) : 
        """computes and stores a histogram for a combination of parameters"""
        i_combo = self._index.get_index(combo)
        if data.size < self.threshold : 
            self.default_contrib[i_combo] = data.size
            H,edges = np.histogram(data,bins=self.default_edges)
            if self.default is None :
                self.default = H
            else : 
                self.default += H
        else : 
            H, edges = np.histogram(data, bins=self.bins)
            self.histograms[i_combo] = (H,edges)
            
    def get_histogram(self, combo) : 
        """returns the histogram associated with the specified combination of parameters"""
        i_combo = tuple(self._index.get_index(combo))
        return self.histograms[i_combo][0]
        
    def get_edges(self, combo) : 
        """returns the bin edges associated with the specified combination of parameters"""
        i_combo = tuple(self._index.get_index(combo))
        return self.histograms[i_combo][1]
        
    def get_combos(self) : 
        """returns the list of parameter combinations present"""
        k = self.histograms.keys()
        combos = [] 
        for cur_index in k : 
            combos.append(self._index.get_unit_val(cur_index))
        return combos
        
            