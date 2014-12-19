import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import qty_index as q


def init_indexers(minmax) :
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
        self._index = init_indexers(minmax)
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
        self.minmax = minmax
        self._index = init_indexers(minmax)
        self.bins = bins
        self.threshold = threshold
        self.default = None
        self.default_minmax = default_minmax
        self.default_contrib = {}
        self.histograms = {}
        self._init_default()
            
    def _init_default(self) : 
        """produce indices for all combos lacking enough data"""
        binsize = float(self.default_minmax[1]-self.default_minmax[0])/self.default_minmax[2]
        self.default_edges = np.arange(self.default_minmax[0],
                                       self.default_minmax[1]+(binsize/2),
                                       binsize)
                                       

    def _add_histo(self, i_combo, H, edges) : 
        self.histograms[i_combo] = (H, edges)  
        
    def _add_default(self, H) : 
        if self.default is None :
            self.default = H
        else : 
            self.default += H
                                        
        
    def put_combo(self, combo, data, units=True) : 
        """computes and stores a histogram for a combination of parameters"""
        if units : 
            i_combo = self._index.get_index(combo)
        else : 
            i_combo = combo
        if data.size < self.threshold : 
            self.default_contrib[i_combo] = data.size
            H,edges = np.histogram(data,bins=self.default_edges)
            self._add_default(H)
        else : 
            H, edges = np.histogram(data, bins=self.bins)
            self._add_histo(i_combo, H, edges)
            
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
        
def save_sparse_histos(sparse_histo, filename) :
    """Save a sparse histogram to a netcdf file"""   
    ncfile = nc.Dataset(filename, 'w')
    
    # create the dimensions
    ncfile.createDimension("bins", sparse_histo.bins)
    ncfile.createDimension("edges", sparse_histo.bins+1)
    ncfile.createDimension("coords", len(sparse_histo.minmax))
    ncfile.createDimension("histograms", len(sparse_histo.histograms)) 
    ncfile.createDimension("axes_definition", 3)
    
    if sparse_histo.default is not None : 
        ncfile.createDimension("default_bins", sparse_histo.default.size)
        ncfile.createDimension("default_edges", sparse_histo.default.size+1)
        # add default_contrib somehow
        
    # create variables
    mmb = ncfile.createVariable("minmaxbin", np.float32, 
                        dimensions=('coords','axes_definition'))
    mmb.long_name = "Minimum, maximum, and binsize values for coordinate axes"
    
    coords = ncfile.createVariable("coordinates", np.int32,
                        dimensions=('histograms', 'coords'))
    coords.long_name = "Location of histogram in parameter space (grid units)"
    
    histos = ncfile.createVariable("histograms", np.float64,
                        dimensions=('histograms','bins'))
    histos.long_name = "histogram bin values" 
    
    edges = ncfile.createVariable("histogram_bin_edges", np.float64,
                        dimensions=('histograms','edges'))
    edges.long_name = "edge values of the histogram bins"
    
    threshold = ncfile.createVariable("threshold", np.float64)
    threshold.long_name = "minimum counts required for a histogram to be individually considered"
    threshold[:] = sparse_histo.threshold
    
    if sparse_histo.default is not None : 
        default_minmax = ncfile.createVariable("default_minmax", np.float64,
                        dimensions=('axes_definition',))
        default_minmax.long_name = "bin definition for default histogram"
        default_minmax[:] = sparse_histo.default_minmax
                            
        default_histo = ncfile.createVariable("default_histogram", np.float64,
                        dimensions=('default_bins',))
        default_histo.long_name = "Histogram for small counts"
        
        default_edges = ncfile.createVariable("default_bin_edges", np.float64,
                        dimensions=('default_edges',))
        default_edges.long_name = "edge values for default histogram"
    
    # populate the axes definition
    for i in range(len(sparse_histo.minmax)): 
        mmb[i,:] = sparse_histo.minmax[i]
        
    
    # populate the histograms
    i_coords = 0
    for k,v in sparse_histo.histograms.iteritems() :
        coords[i_coords,:] = k
        histos[i_coords,:] = v[0]
        edges[i_coords,:]  = v[1]
        i_coords += 1
    
    # close   
    ncfile.close()   
    
def load_sparse_histos(histofile) :
    ncfile = nc.Dataset(histofile) 
    
    default_minmax = ncfile.variables['default_minmax'][:]
    minmax = ncfile.variables['minmaxbin'][:]
    bins = len(ncfile.dimensions['bins'])
    threshold = ncfile.variables['threshold'][:]
    
    shisto = SparseKeyedHistogram(minmax=minmax, default_minmax=default_minmax,
                        bins=bins, threshold=threshold)
                
    # populate the default histogram
    shisto._add_default(ncfile.variables['default_histogram'][:])
    # default histogram edges should already be populated
    
    # populate the histogram dictionary.
    i_combos = shisto.variables['coordinates'][:]
    histos   = shisto.variables['histograms'][:]
    edges    = shisto.variables['histogram_bin_edges'][:]
    for i_histo in range(len(ncfile.dimensions['histograms'])) :         
        shisto._add_histo(i_combos[i_histo,:], histos[i_histo,:], edges[i_histo,:])
    
    ncfile.close()
    return shisto   