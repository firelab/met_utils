"""Manipulates burned area

Contains code primarily intended to accumulate a shapefile of burned area
points onto a low-resolution grid.
"""

import qty_index as q
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import astropy.units as u
import astropy.time as time
import aggregator as agg
import reduce_var as rv
import dateutil.parser as dup
import pandas as pd
import orchidee_indices as oi
import accum_hist as ah
import trend
import collections
import geo_ca as gca
from osgeo import ogr



# MOD12 landcover classification codes, broken down into 
# forest and not forest.
FOREST_LC = range(1,6)
NONFOREST_LC = range(6,11)

def landcover_classification(v) : 
    """returns indices of breakpoints between landcover classification

    Given a sorted coordinate variable v, this function returns a 
    four element sequence which gives the indices of the breakpoints
    between landcover classes. The landcover classes are "forest", "nonforest",
    and "other". The codes for these classes are given in
    FOREST_LC and NONFOREST_LC, with a third class constructed from
    landcover codes greater than the last value contained in NONFOREST_LC.
    """
    edges = [ 0 ]
    state = 'f'
    for i in range(len(v)) : 
        if state == 'f' : 
            if v[i] in FOREST_LC : continue
            edges.append(i)
            state = 'nf'
        if state == 'nf' : 
            if v[i] in NONFOREST_LC : continue
            edges.extend( (i, len(v)) )
            break
    if len(edges) == 2 :
        edges.extend( [len(v)]*2 ) 
    return edges

class BurnedAreaShapefile ( object )  :
    """encapsulates some characteristics of the Burned Area Shapefiles"""
    def __init__(self, shapefilename) : 
        # open the shapefile & get layer
        driver = ogr.GetDriverByName('ESRI Shapefile')
        self.dataSource = driver.Open(shapefilename, 0) # 0 means read-only. 1 means writeable.
        self.layer = self.dataSource.GetLayer()
        self.layerName = self.layer.GetName()
        
        self._get_landcover_codes() 
        
    def _get_landcover_codes(self) :
        sql =  "SELECT DISTINCT LANDCOVER FROM '%s' ORDER BY LANDCOVER" % self.layerName
        results = self.dataSource.ExecuteSQL(sql) 
        
        landcover_codes =  []
        for feature in results : 
            landcover_codes.append(feature.GetField("LANDCOVER"))
            
        self.dataSource.ReleaseResultSet(results)
        
        self.landcover_codes = landcover_codes
        
    def query_ascending_date(self) : 
        """returns features in ascending order by date
        
        destroy the returned layed with release_results()"""
        sql = "SELECT * FROM '%s' ORDER BY DATE" % self.layerName
        return self.dataSource.ExecuteSQL(sql)
        
    def release_results(self, query) : 
        """call this to dispose of results from query"""
        self.dataSource.ReleaseResultSet(query)
        
class BurnedAreaCounts (agg.NetCDFTemplate) : 
    def __init__(self, template, ncfile, landcover_codes, year) :     
        # open file / copy info from template
        super(BurnedAreaCounts, self).__init__(template,ncfile)
        self.copyVariable('nav_lat')
        self.copyVariable('nav_lon')
        self.copyDimension('days')
        self.createDimension('landcover', len(landcover_codes))
        srcvar = self._ncfile.variables
        dims   = self._ncfile.dimensions
        self.x = len(dims['x'])
        self.y = len(dims['y'])
        self.days = len(dims['days'])
        self.landcover = len(landcover_codes)
        
        # make coordinate variable for landcover
        self.add_variable(np.array(landcover_codes, dtype=np.int8),'landcover',('landcover',))

        # make indexers for various axes
        delta_lat=  (srcvar['nav_lat'][1,0]-srcvar['nav_lat'][0,0])*u.deg
        delta_lon=  (srcvar['nav_lon'][0,1]-srcvar['nav_lon'][0,0])*u.deg
        self.lat_idx = q.LinearSamplingFunction(1./delta_lat, x_zero=srcvar['nav_lat'][0,0]*u.deg)
        self.lon_idx = q.LinearSamplingFunction(1./delta_lon, x_zero=srcvar['nav_lon'][0,0]*u.deg)    
        self.time_idx = q.TimeSinceEpochFunction(1*u.day, time.Time('%d:001'%year,format='yday'))
        self.lc_idx = q.CoordinateVariableSamplingFunction(landcover_codes)
        
        #create the netcdf variable
        self.count = self.create_variable('count', ('y','x','landcover','days'), np.int16,
                        chunk=(self.y, self.x, self.landcover,1))
        self.last_day = -1
        
    def new_cache(self) : 
        """creates a new cache for a single day's data"""
        cache = np.zeros( (self.y, self.x, self.landcover), dtype=np.int16)
        idx   = q.OrthoIndexer( [self.lat_idx,self.lon_idx, self.lc_idx])
        return q.UnitIndexNdArray(cache, idx)
        
    def put_cache(self, cache, day) : 
        """stores the cache for a day's data into the file
        
        If a day (or more) has been skipped, write zeros to the 
        intervening days...
        """
        print day
        if self.last_day != (day-1) : 
            zeros = self.new_cache()
            for d in range(self.last_day+1, day) : 
                print d
                self.count[...,d] = zeros.array
        self.last_day = day

        # Make sure we're not trying to add data past the end 
        # of what was in the template netCDF file.
        if day < self.days : 
            self.count[...,day] = cache.array
        

def ba_year(year, template, ncfile, shapefile)  :
    """creates a burned area netcdf file from the template
    
    The template provided should be a netcdf file of indices"""
    
    shp = BurnedAreaShapefile(shapefile)
    bac = BurnedAreaCounts(template, ncfile, shp.landcover_codes, year)    
    
    day = -1
    cache = None
    layer = shp.query_ascending_date()      
    for feature in layer :
        # first, compute the day index of this new item 
        t = feature.GetField("DATE")
        dt = dup.parse(t)
        # convert to UTC if necessary
        if dt.tzinfo is not None: 
            dt = (dt - dt.utcoffset()).replace(tzinfo=None)
            if dt.timetuple()[7] >= 366 : 
                continue
        t_obj = time.Time(dt)
        i_day = bac.time_idx.get_index(t_obj).value
        
        # if new day, advance to the next day, get a new cache
        if i_day != day : 
            if day != -1 : 
                bac.put_cache(cache, day)
            day = i_day
            cache = bac.new_cache()
        
        # accumulate in the cache        
        geom = feature.GetGeometryRef()
        if geom.GetGeometryName() == 'POLYGON' : 
            loc = geom.GetGeometryRef(0).GetPoint(0) * u.deg
            lc_code = feature.GetField('LANDCOVER')
        
            # increment the correct counter cell
            cache.inc( (loc[1],loc[0],lc_code))
    
    # wrap up
    bac.put_cache(cache, day)
    shp.release_results(layer)    
    bac.close()
    
REDUCE_MONTHLY = -1
def ba_compare_year(indicesfile, bafile, outfile=None, indicesvarnames= None, support=None, reduction=None) : 
    """collates the various indices with burned area counts"""

    # interesting variables from the indicesfile
    last_val_reduce = [ ] 
    if indicesvarnames is None : 
        indicesvarnames = ['gsi_avg','fm1000','fm100','fm10','fm1','dd','t_max']
    if 'dd' in indicesvarnames : 
        last_val_reduce = ['dd']

    indices = nc.Dataset(indicesfile)
    indicesvars = [indices.variables[v] for v in indicesvarnames]
    ba = nc.Dataset(bafile)
    count = ba.variables['count']

    if support is not None : 
        s = nc.Dataset(support)
        supportvars = list(s.variables.keys())
        supportvars.remove('land')
        indicesvarnames.extend(supportvars)
        indicesvars.extend([s.variables[v] for v in supportvars])
        last_val_reduce.extend(supportvars)
    
    # workaround: bug in index calculator does not calculate last day.
    time_samples = range(1,len(ba.dimensions['days']) - 1)
    if reduction is not None : 
        if reduction == REDUCE_MONTHLY  : 
            grid_reducer = rv.monthly_aggregator(count.shape, 3)
            cmp_reducer  = rv.monthly_aggregator(indicesvars[0].shape,0)
            grid_reducer.cutpoints[0]=1
            cmp_reducer.cutpoints[1]=1
        else : 
            grid_reducer = rv.ReduceVar(count.shape, 3, reduction)
            cmp_reducer  = rv.ReduceVar(indicesvars[0].shape, 0, reduction)
        time_samples = range(grid_reducer.reduced)

    ca = trend.CompressedAxes(indices, 'land')

    alldata = []
    days = [] 

    for i_time in time_samples : 
        day_data = [] 
        active_lc = [] 
        

        if reduction is None :
            count_slice = count[...,i_time]
        else :
            count_slice = np.array(grid_reducer.sum(i_time, count))
            
        for lc in range(len(ba.dimensions['landcover'])) : 
            
            # compress the count
            lc_count = ca.compress(count_slice[:,:,lc])
    
            # find nonzero counts
            i_nonzero = ma.nonzero(lc_count)
    
            if len(i_nonzero[0]) > 0 : 
                # construct dataframe for this landcover code
                lc_data = {"BA Count" : lc_count[i_nonzero]}
    
                for n,v in zip(indicesvarnames,indicesvars) : 
                    # reduce variable if necessary
                    if reduction is None: 
                        day_v = v[i_time,:]
                    else : 
                        # the last value of the dry day sequence is 
                        # representative of the reduced time period
                        if n in last_val_reduce : 
                            day_v = cmp_reducer.last_val(i_time, v)
                        else : 
                            day_v = cmp_reducer.mean(i_time, v)

                    # add a column for the current index    
                    lc_data[n] = day_v[i_nonzero]
    
                day_data.append( pd.DataFrame( lc_data ) )
                active_lc.append(ba.variables['landcover'][lc])
            
        if len(day_data) > 0 : 
            alldata.append(pd.concat(day_data, keys=active_lc)) 
            days.append(i_time)

    all_data_frame = pd.concat(alldata, keys=days)

    if outfile is not None : 
        all_data_frame.to_csv(outfile)

    return all_data_frame

def ba_ratio_histograms(ba_files, ind_files, indices_names,minmax) :
    """computes histogram of ratio of MODIS BA to 0.5x0.5 deg occurrence
    
    Considering each day an independent measurement of the entire study area, 
    the ratio of total MODIS BA counts to total 0.5x0.5 deg cells is computed 
    for each parameter bin. Each parameter bin gets at most one observation per 
    day, and this observation embodies all 0.5x0.5deg cells in that bin for that
    day.
    """ 
    num_years = len(ind_files)
    max_days = 365
    histo_shape = zip(*minmax)[2]
    ratio_shape = histo_shape + (max_days,num_years)
    ratios = ma.masked_all(ratio_shape)
    halfdeg_counts = ma.masked_all(ratio_shape)
    
    
    ca = gca.GeoCompressedAxes(ind_files[0], 'land') 
    ca.set_clip_box(42.5, 66.5, 22, 130)
    
    for i_year in range(len(ind_files)) : 
        indfile = ind_files[i_year]
        bafile  = ba_files[i_year]
        count   = bafile.variables['count']
        timelim = len(indfile.dimensions['days'])-1
        filevars = [ indfile.variables[iname] for iname in indices_names ] 
        for i_day in range(10,timelim) : 
            print i_day
            day_data = [ f[i_day,:] for f in filevars ]
            i_conditions = zip(*day_data)
            ba_day = count[...,i_day]
            ba_total = np.sum(ba_day, axis=2)
            ba_total_cmp = ca.compress(ba_total)
            
            
            # per bin ba totals (units of modis pixels)
            burned_total = ah.AccumulatingHistogramdd(minmax=minmax)
            for i_tot,ba_tot in enumerate(ba_total_cmp) : 
                if ba_tot is ma.masked  :
                    continue
                if ba_tot > 0 : 
                    burned_total.put_record(i_conditions[i_tot], weight=ba_tot)
            
            # per bin occurrence totals (units of 0.5 deg cells)
            occurrence = ah.AccumulatingHistogramdd(minmax=minmax)
            for i_window,mask in enumerate(ca.get_vec_mask()) : 
                if not mask : 
                    occurrence.put_record(i_conditions[i_window])
            
            # calculate ratio
            i_occurrence = np.where(occurrence.H > 0)
            num_occurrence = len(i_occurrence[0])
            i_occ_oneday = i_occurrence + ( np.array([i_day]*num_occurrence), np.array([i_year]*num_occurrence))
            ratios[i_occ_oneday] = burned_total.H[i_occurrence]/occurrence.H[i_occurrence]
            halfdeg_counts[...,i_day,i_year] = occurrence.H

    ratio_histogram = compute_ratio_histo(ratios, minmax)
            
    return (ratios, halfdeg_counts, ratio_histogram)

def compute_ratio_histo(ratios, minmax, min_bins=5): 
    # the result 
    ratio_histogram = ah.SparseKeyedHistogram(minmax=minmax)

    
    # iterate over all the bins, extracting the time series and adding to the 
    # sparse histogram
    just_the_bins = ratios[...,0,0]
    it = np.nditer(just_the_bins, flags=['multi_index'])
    while not it.finished : 
        combo = it.multi_index
        i_extract = combo + (Ellipsis,)
        bin_data = ratios[i_extract]
        bin_data = bin_data.compressed()
        if bin_data.size > 0 : 
            ratio_histogram.put_combo(combo, bin_data, units=False,min_bins=min_bins)
        it.iternext()

    return ratio_histogram

def calc_geog_mask(ca, bafile, geog_box) : 
    """calculates a 1d mask based on a tuple representing a bounding box
    expressed in lat/lon
    
    The lookup process to convert geographic lat/lon to 2d index values
    requires that the provided value actually be in the array. No in-betweens.
    
    The returned 1d mask is a boolean array where False indicates the pixel is
    included in the ROI (not-masked), and True indicates the pixel is outside
    the ROI.
    """
    #
    # TODO: This can be improved by not performing a lookup on the cell values.
    #   This does flip the boolean "sense" of the result (included pixels are
    #   True). 
    # -    mask_2d = (nav_lat >= min_lat) & \
    # -              (nav_lat <= max_lat) & \
    # -              (nav_lon >= min_lon) & \
    # -              (nav_lon <= max_lon)
    # -    mask_1d = ca.compress(mask_2d)
    
    nav_lat = bafile.variables['nav_lat'][:]
    nav_lon = bafile.variables['nav_lon'][:]
    
    min_lon = geog_box[0]
    max_lon = geog_box[1]
    min_lat = geog_box[2]
    max_lat = geog_box[3]

    # inverting the logic in the above comment yields a 2d mask where 
    # included pixels get a False value (not-masked)
    mask_2d = (min_lat > nav_lat) | \
              (max_lat < nav_lat) | \
              (min_lon > nav_lon) | \
              (max_lon < nav_lon)
    
    return ca.compress(mask_2d)
    
            

def ba_multifile_histograms(ba_files, ind_files, indices_names,minmax, 
                             day_range=None, geog_box=None) : 
    """calculates combined index-oriented and MODIS BA oriented histograms
    
    The user can specify a day of year range and geography box to limit the 
    data. Geography box is specified as a tuple: (lon_min, lon_max, lat_min,
    lat_max). 
    
    Computes and returns nine histograms using the minmax description 
    provided. Five histograms involve only the indices, which are assumed 
    to be computed at a coarse resolution such as 0.5 deg by 0.5 deg. 
    These histograms are computed with a uniform weight of 1 for every 
    occurrence. One represents all observed combinations of indices,
    the other represents all combinations of indices observed to contain 
    some level of burning. From these, unburned area for each combination
    of indices can be derived.

    The remaining four histograms represent high resolution burned area data
    aggregated to the coarse resolution grid. It is assumed that landcover 
    information is only available for the high resolution data. Separate
    histograms are calculated for groups of landcover codes representing
    forest, non-forest, and "other". These histograms, as well as a "total"
    histogram, are weighted by the burned area observed to occur at each 
    combination of indices. These four histograms represent only burned
    area, and do not contain information from which unburned area may be 
    derived.
    """
    one_day = len(ind_files[0].dimensions['land'])

    # these count 0.5 x 0.5 degree cells
    occurrence = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int32)
    burned_occurrence = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int32)
    burned_forest_occ = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int32)
    burned_not_forest_occ = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int32)
    burned_other_occ      = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int32)

    # these four count individual modis detections
    burned_forest = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int64) 
    burned_not_forest = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int64)
    burned_other = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int64)
    burned_total = ah.AccumulatingHistogramdd(minmax=minmax, dtype=np.int64)

    ca = trend.CompressedAxes(ind_files[0], 'land') 
    
    if geog_box is not None : 
        geog_mask = calc_geog_mask(ca, ba_files[0], geog_box)
    else : 
        geog_mask = np.zeros( (one_day,), dtype=np.bool)

    for i_year in range(len(ind_files)) : 
        # fetch the correct file handles for this year
        indfile = ind_files[i_year]
        bafile  = ba_files[i_year]
        
        # get BA handle and initialize an object to aggregate BA by
        # landcover type
        count   = bafile.variables['count']
        lc_edges = landcover_classification(bafile.variables['landcover'][:])
        lc_type = rv.CutpointReduceVar(count.shape[:-1], 2, lc_edges)
        
        # get number of samples along the time dimension
        timelim = len(indfile.dimensions['days'])-1
        timerange = range(1,timelim)
        if day_range is not None : 
            timerange = range(day_range.start, day_range.stop)
        
        # get variable references for each index
        filevars = [ indfile.variables[iname] for iname in indices_names ] 
        
        for i_day in timerange : 
            print i_year, i_day
            # grab one day's worth of data out of each index variable
            day_data = [ f[i_day,:] for f in filevars ]
            
            # grab one day's worth of data 
            ba_day = count[...,i_day]
            
            # aggregate the data
            ba_forest = lc_type.sum(0,ba_day)
            ba_nonforest = lc_type.sum(1,ba_day)
            ba_other     = lc_type.sum(2,ba_day)
            
            # compress the aggregated data into the 1D land array
            ba_forest_cmp = ca.compress(ba_forest)
            ba_nonforest_cmp = ca.compress(ba_nonforest)
            ba_other_cmp = ca.compress(ba_other)
            
            # construct an array of records for all land pixels in a single 
            # day. Also construct a mask which can be used to pull data from
            # the weights arrays
            records = ma.zeros( (one_day, len(day_data)))
            
            # compile all the records for a single day (column-wise)
            for i_data in range(len(day_data)):
                records[:,i_data] = day_data[i_data]
                
            # filter out pixels where any of the indices are missing. (row-wise)    
            # Merge in the geographic filter.
            if len(day_data) > 1 : 
                land_data = np.any(records.mask, axis=1)
            else : 
                land_data = records.mask.squeeze()
            land_data = np.logical_not(land_data | geog_mask)
                
            # extract out just the records with data
            records = records[land_data,:]    
            
            occurrence.put_batch(records)
            burned_weight= np.zeros( (np.count_nonzero(land_data),))
            
            # for each of the histograms which count only burned area, 
            # extract those records with nonzero burned area and 
            # submit them as a batch to the relevant histogram.
            ba = ba_forest_cmp[land_data]
            if np.count_nonzero(ba) > 0 : 
                idx = np.where( ba != 0)
                rec = records[idx,:].squeeze(axis=(0,))
                burned_forest.put_batch(rec, weights=ba[idx])
                burned_forest_occ.put_batch(rec)
                burned_weight += ba
            
            ba = ba_nonforest_cmp[land_data]
            if np.count_nonzero(ba) > 0 : 
                idx = np.where( ba != 0)
                rec = records[idx,:].squeeze(axis=(0,))
                burned_not_forest.put_batch(rec, weights=ba[idx])
                burned_not_forest_occ.put_batch(rec)
                burned_weight += ba
            
            ba = ba_other_cmp[land_data]
            if np.count_nonzero(ba) > 0 : 
                idx = np.where( ba != 0)
                rec = records[idx,:].squeeze(axis=(0,))
                burned_other.put_batch(rec, weights=ba[idx])
                burned_other_occ.put_batch(rec)
                burned_weight += ba
            
            ba = burned_weight
            if np.count_nonzero(ba) > 0 : 
                idx = np.where( ba != 0)
                rec = records[idx,:].squeeze(axis=(0,))
                burned_total.put_batch(rec, weights=ba[idx])
                burned_occurrence.put_batch(rec)
                
    return (occurrence, burned_occurrence, 
             burned_forest, burned_forest_occ, 
             burned_not_forest, burned_not_forest_occ,
             burned_other, burned_other_occ, burned_total)

def create_multiyear_histogram_file(outfile, ind_names, minmax) : 
    """creates a histogram file with dimensions and coordinate variables"""
    ofile = nc.Dataset(outfile, 'w')
        
    # create dimensions
    for i_ind, indname in enumerate(ind_names) : 
        cur_min, cur_max, cur_bins = minmax[i_ind]
        ofile.createDimension(indname, cur_bins)
        cv = ofile.createVariable(indname, np.float64, dimensions=(indname,)) 
        binsize    = float(cur_max - cur_min)/cur_bins
        cv[:] = np.arange(cur_min, cur_max, binsize)
        cv.binsize = binsize
        
    return ofile
    

def write_multiyear_histogram_file(outfile, histos, ind_names, minmax, 
        day_range=None, geog_box=None)  :
    """write a multiyear histogram netcdf file"""

    # create file, dimensions, variables
    ofile = create_multiyear_histogram_file(outfile, ind_names, minmax)

    # record the doy range used to generate this data.
    if day_range is not None : 
        ofile.start_day = day_range.start
        ofile.end_day   = day_range.stop-1

    # record the geographic bounding box used to generate this data
    if geog_box is not None : 
        ofile.min_lon = geog_box[0]
        ofile.max_lon = geog_box[1]
        ofile.min_lat = geog_box[2]
        ofile.max_lat = geog_box[3]
    
    # store variables
    names = [ 'occurrence', 'burned_occurrence', 
              'burned_forest', 'burned_forest_occ', 
              'burned_not_forest', 'burned_not_forest_occ',
              'burned_other', 'burned_other_occ', 'burned_total'] 
    # things which count 0.5 deg cells are int32, things which count 
    # MODIS cells are int64
    types = [ np.int32, np.int32,
              np.int64, np.int32,
              np.int64, np.int32,
              np.int64, np.int32, np.int64 ]
    for name, hist, t in zip(names, histos, types) : 
        v = ofile.createVariable(name, t, ind_names) 
        v[:] = hist.H 
        v.count = hist.count
        v.total = hist.total

    # close outfile
    ofile.close()

def ba_multiyear_histogram(years, ba_template, ind_template, ind_names, 
                outfile=None, bins=10, minmaxyears=None) :
    """computes multiyear histograms and stores in a netcdf file."""

    # open netcdf files
    bafiles = [ ]
    indfiles = [ ] 
    for y in years : 
        bafiles.append(nc.Dataset(ba_template % y))
        indfiles.append(nc.Dataset(ind_template % y))

    # compute min/max
    if minmaxyears is None :
        minmax = oi.multifile_minmax(indfiles, ind_names)
    else :
        minmax = oi.multifile_minmax(ind_template, ind_names, years=minmaxyears)
        
    if not ('__iter__' in dir(bins)) : 
        bins = [ bins ] * len(ind_names)
    minmax = zip(minmax[0], minmax[1], bins)

    # compute histogram
    histos = ba_multifile_histograms(bafiles, indfiles, ind_names, minmax)
    
    # write output
    if outfile is not None : 
        write_multiyear_histogram_file(outfile, histos, ind_names, minmax)

    # close netcdf files
    for i_files in range(len(years)) : 
        bafiles[i_files].close()
        indfiles[i_files].close()

    return histos
    
# a variant specialized to calculate percentile & univariate histograms
# this puts the histogram of each index in a separate file.
# ba_template, ind_template, and outfile should all be templates. The outfile
# template should expect to receive an index name.
def ba_multiyear_pct_histogram(years, ba_template, ind_template, ind_names, 
                outfile=None, day_range=None, geog_box=None) :
    """computes multiyear histograms and stores in a netcdf file."""

    # open netcdf files
    bafiles = [ ]
    indfiles = [ ] 
    for y in years : 
        bafiles.append(nc.Dataset(ba_template % y))
        indfiles.append(nc.Dataset(ind_template % y))

    # compute min/max
    minmax = [[0, 100, 101]]

    for ind in ind_names : 
        ind_list = [ ind ]
        # compute histogram
        histos = ba_multifile_histograms(bafiles, indfiles, ind_list, minmax,
                     day_range, geog_box)
    
        # write output
        if outfile is not None : 
            write_multiyear_histogram_file(outfile%ind, histos, ind_list, 
                     minmax, day_range, geog_box)

    # close netcdf files
    for i_files in range(len(years)) : 
        bafiles[i_files].close()
        indfiles[i_files].close()

    return histos
    
def ba_multiyear_add_ratios(filename) : 
    """adds ratios to a file created with write_multiyear_histogram_file() 
    
    The ratios are computed element-wise with the "occurrence" variable in
    the denominator."""
    names = [ 'burned_occurrence', 
              'burned_forest', 'burned_forest_occ', 
              'burned_not_forest', 'burned_not_forest_occ',
              'burned_other', 'burned_other_occ', 'burned_total'] 
              
    ncfile = nc.Dataset(filename, mode='r+')
    occ = np.array(ncfile.variables['occurrence'][:], dtype=np.float)
    occdims = ncfile.variables['occurrence'].dimensions
    
    for v in names : 
        ratio = ncfile.variables[v][:] / occ
        newname = 'ratio_{:s}'.format(v)
        newvar = ncfile.createVariable(newname, ratio.dtype, 
                                    occdims, fill_value=-1)
        newvar[:] = ratio
        
    ncfile.close()
        
        


def ba_univ_agg_multiyear_histogram(csv_files, years, agg_col, bins=range(0,102), 
         weight_col=None) : 
    """Univariate aggregation of dataset onto a histogram.
    
    Aggregates a single column of a series of datasets into a histogram. The 
    filename pattern is specified in csv_files. The input files are expected to
    be compatible with the CSV files produced by ba_compare_year(). The name of 
    the column to aggregate is specified in agg_col. By default, 101 integer 
    bins from 0 to 101 are specified (initial application is to support 
    percentile binning). A straight histogram is calculated by default, but 
    naming a "weight column" allows for the computation of the weighted histogram.
    
    All of the CSV files must have a header row containing the column names as 
    row zero, and must possess the column named in agg_col and (if specified) 
    in weight_col.
    """
    if weight_col is None: 
        acc_type = np.int
    else :
        acc_type = np.float
        
    accumulator = np.zeros( ( len(bins)-1, ), dtype = acc_type)
    
    for y in years : 
        ds = pd.read_csv(csv_files % y, header=0)
        if weight_col is None:  
            cur, bins = np.histogram(ds[agg_col], bins=bins)
        else : 
            cur, bins = np.histogram(ds[agg_col], bins=bins, weights=ds[weight_col])
        accumulator += cur
        
    return accumulator
        

def select_data(dataframe, names, i_count, indexer, dim_bins, lc_codes=None) :
    """Selects rows in a dataframe based on criteria in multiple columns.
    
    This is largely intended to select data records contained within an 
    "nd bin" from a dataset. The nd bin's nd index is specified by "i_count".
    Each row represents a coordinate vector, where the columns represent the 
    ordinate along a single axis.
    """
    u_lower = indexer.get_unit_val(i_count)
    u_upper = indexer.get_unit_val(np.add(i_count,1))
    pegged = [i == d for i,d in zip(i_count,dim_bins)]
    
    # initialize to "everything" or "everything in a set of landcover codes"
    if lc_codes is None :
        i_data = np.ones( (dataframe.shape[0],), dtype=np.bool)
    else : 
        i_data = [dataframe.ix[i,1] in lc_codes for i in range(dataframe.shape[0])]
    for low,high,name,p in zip(u_lower, u_upper, names, pegged) :
        if not pegged : 
            i_cur = np.logical_and(dataframe[name]>=low, dataframe[name]<high)
        else : 
            i_cur = np.logical_and(dataframe[name]>=low, dataframe[name]<=high)
        i_data = np.logical_and(i_data, i_cur)
        
    return dataframe[i_data]

def read_multiyear_minmax( bahist, dimnames ) : 

    mmb = [] 
    binsizes = [ ]
    for dimname in dimnames: 
        dim = bahist.dimensions[dimname]
        cv = bahist.variables[dimname][:]
        binsizes.append( bahist.variables[dimname].binsize )
        mmb.append( (cv[0], cv[-1]+binsizes[-1], len(dim)))

    return mmb, binsizes

def sparse_multiyear_histogram(years, csv_template, bahistfile, 
                            count_threshold=50, bins=25, out_template=None) : 
    """computes and optionally saves sparse histograms of MODIS BA counts"""
    # open the ba histogram file
    bahist = nc.Dataset(bahistfile)
    counts = bahist.variables['burned_total']
    
    # read all csv files and concatenate
    file_list = []
    for y in years :  
        file_list.append(pd.read_csv(csv_template % y))
    compare = pd.concat(file_list)
    compare = compare[ np.logical_and(compare.icol(0)>=10,compare.icol(0)<364) ] 
    
    # get min/max/bin from multiyear histogram file
    mmb, binsizes = read_multiyear_minmax(bahist,counts.dimensions)
        
    # create an indexer
    index = ah.init_indexers(mmb) 
    
    # strip out geometry
    dim_bins = [m[2] for m in mmb]   
    
    # create sparse histograms
    shisto_forest = ah.SparseKeyedHistogram(minmax=mmb, threshold=count_threshold,
                           bins=bins, default_minmax=(1,count_threshold,count_threshold-1))
    shisto_not_forest = ah.SparseKeyedHistogram(minmax=mmb, threshold=count_threshold,
                           bins=bins, default_minmax=(1,count_threshold,count_threshold-1))
    shisto_total = ah.SparseKeyedHistogram(minmax=mmb, threshold=count_threshold,
                           bins=bins, default_minmax=(1,count_threshold,count_threshold-1))

                           

    # loop through all bins with nonzero data
    i_nonzero = np.where( counts[:]>0 )
    for i_bin in zip(*i_nonzero) : 
        total = select_data(compare, counts.dimensions, i_bin, index, dim_bins)
        forest = total[ total.ix[:,1].isin(FOREST_LC) ]
        not_forest = total [ total.ix[:,1].isin(NONFOREST_LC) ]

        shisto_forest.put_combo(i_bin, forest['BA Count'], units=False)
        shisto_not_forest.put_combo(i_bin, not_forest["BA Count"], units=False)
        shisto_total.put_combo(i_bin, total['BA Count'], units=False)
        
    # save file if filename template specified
    if out_template is not None : 
        ah.save_sparse_histos(shisto_total, out_template%'total')
        ah.save_sparse_histos(shisto_forest, out_template%'forest')
        ah.save_sparse_histos(shisto_not_forest, out_template%'not_forest')
        
    bahist.close()
    
    return (shisto_total, shisto_forest, shisto_not_forest)
                         
def write_raw_ratio_file(outfile, ratios, halfdeg_counts, ind_names, minmax) : 
    
    # create file/dimensions/coordinate variables
    ofile = create_multiyear_histogram_file(outfile, ind_names, minmax) 
    
    # add dimensions for days and years
    ofile.createDimension('day_of_year', ratios.shape[-2])
    ofile.createDimension('year', ratios.shape[-1])
    vardims  =  ind_names + ['day_of_year','year']
    
    # save the ratios
    rat = ofile.createVariable('raw_ratios', np.float64, dimensions=vardims)
    rat[:] = ratios
    
    # save the occurrence counts
    ho = ofile.createVariable('indices_occurrence', np.float64, dimensions=vardims)
    ho[:] = halfdeg_counts
    
    ofile.close() 
    
def ba_multiyear_ratios(years, ba_template, ind_template, ind_names, 
                histo_outfile=None, ratio_outfile=None, bins=10, minmaxyears=None) :
    """computes multiyear histograms and stores in a netcdf file."""

    # open netcdf files
    bafiles = [ ]
    indfiles = [ ] 
    for y in years : 
        bafiles.append(nc.Dataset(ba_template % y))
        indfiles.append(nc.Dataset(ind_template % y))

    # compute min/max
    if minmaxyears is None :
        minmax = oi.multifile_minmax(indfiles, ind_names)
    else :
        minmax = oi.multifile_minmax(ind_template, ind_names, years=minmaxyears)
        
    if not ('__iter__' in dir(bins)) : 
        bins = [ bins ] * len(ind_names)
    minmax = zip(minmax[0], minmax[1], bins)

    # compute histogram
    ratios, halfdeg_counts, histo = ba_ratio_histograms(bafiles, indfiles, ind_names, minmax)
    
    # write output
    if histo_outfile is not None : 
        ah.save_sparse_histos(histo, histo_outfile)
    if ratio_outfile is not None: 
        write_raw_ratio_file(ratio_outfile, ratios, halfdeg_counts, ind_names, minmax)

    # close netcdf files
    for i_files in range(len(years)) : 
        bafiles[i_files].close()
        indfiles[i_files].close()

    return ratios, histo

# add a function which just aggregates the values along the time axis 
# in a netCDF file, producing another, compatible netCDF file for downstream 
# processing.
def aggregate(infile, outfile, reduction, variables=None, 
      agg_methods=rv.ReduceVar.REDUCE_MEAN, 
      agg_dim='days') : 
    """copy named variables and aggregate in the specified manner
    
    Copy infile to outfile, aggregating the named variables by the specified
    "reduction factor" using agg_methods to produce the representative value
    in "outfile"."""
    in_ds = nc.Dataset(infile)
    
    # if the user did not specify which variables to reduce, 
    # guess that they want everything except coordinate variables.
    if variables is None: 
        variables = list(in_ds.variables.keys())
        for d in in_ds.dimensions.keys() : 
            variables.remove(d)
        if 'nav_lat' in variables : 
            variables.remove('nav_lat')
        if 'nav_lon' in variables :
            variables.remove('nav_lon')
            
    # set up the "ReduceVar" aggregator
    # assume that all variables have same dimensions.
    v = in_ds.variables[variables[0]]
    variable_shape = v.shape
    variable_dims  = v.dimensions
    i_agg = variable_dims.index(agg_dim)
    if reduction == REDUCE_MONTHLY : 
        aggregator = rv.monthly_aggregator(variable_shape, i_agg) 
    else : 
        aggregator = rv.ReduceVar(variable_shape, i_agg, reduction)
        
    # figure out the shape of the output array 
    output_shape = list(variable_shape)
    output_shape[i_agg] = aggregator.reduced
    
    # create the output file
    out_agg = agg.NetCDFTemplate(infile, outfile)
    
    # don't let the template copy the "aggregate" dimension to the new file!
    out_agg.createDimension(agg_dim, aggregator.reduced)
    
    # copy the "navigation" variables
    out_agg.copyVariable('nav_lat')
    out_agg.copyVariable('nav_lon')
    
    # expand agg_methods if necessary
    if not isinstance(agg_methods, collections.Sequence) : 
        agg_methods = [agg_methods] * len(variables)

    # prepare an index to write the output
    out_slice = [ slice(None,None,None) ] * len(variable_shape)
    
    # loop over the variables        
    for varname, agg_method in zip(variables, agg_methods) : 
        v = in_ds.variables[varname]
        fill_value = getattr(v, '_FillValue', None)
        out_v = out_agg.create_variable(varname, v.dimensions, 
                       v.dtype, fill=fill_value)

        # loop over each reduced index        
        for reduced_i in range(aggregator.reduced) : 
            out_slice[i_agg] = reduced_i
            out_v[out_slice] = aggregator.reduce(agg_method, reduced_i, v)
            
    out_agg.close()
    in_ds.close()
