import trend
import qty_index as qi
import numpy as np
import numpy.ma as ma


class GeoCompressedAxes ( trend.CompressedAxes )  :
    """compressed axes object which interprets geospatial coordinates
    
    This object is capable of converting between a 2d grid and a 1d compressed
    vector representation, taking into account the geospatial coordinates of 
    each cell. This makes it possible to mask the 1d or 2d array by lat/lon
    boundaries.
    """
    def __init__(self, dataset, c_dim, format='F') : 
        super(GeoCompressedAxes, self).__init__(dataset, c_dim, format)
        self._init_orchidee_geo(dataset)
        
    def _init_orchidee_geo(self, dataset) : 
        """initializes the geospatial indexer from nav_lat and nav_lon in the dataset"""
        
        lats = dataset.variables['nav_lat'][:,0]
        lons = dataset.variables['nav_lon'][0,:]
        
        delta_lat = lats[1] - lats[0]
        delta_lon = lons[1] - lons[0]
        
        lat_samp = qi.LinearSamplingFunction(1./delta_lat, x_zero=lats[0])
        lon_samp = qi.LinearSamplingFunction(1./delta_lon, x_zero=lons[0])
        
        self._samp = qi.OrthoIndexer([lat_samp,lon_samp])
        
        
        
    def set_clip_box(self, min_lat, max_lat, min_lon, max_lon) : 
        """set this object's mask to the provided lat/lon box"""
        mask = np.ones(self._dimshape, dtype=np.bool)
        
        i_min_lat, i_min_lon = self._samp.get_index( (min_lat, min_lon))
        i_max_lat, i_max_lon = self._samp.get_index( (max_lat, max_lon))
        
        # unmask the inside of the window
        mask[i_min_lat:i_max_lat+1, i_min_lon:i_max_lon+1] = False
        self.set_grid_mask(mask)
        
        
        