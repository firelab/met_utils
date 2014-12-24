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
    

def ba_compare_year(indicesfile, bafile, outfile=None, support=None, reduction=None) : 
    """collates the various indices with burned area counts"""

    # interesting variables from the indicesfile
    indicesvarnames = ['gsi_avg','fm1000','fm100','fm10','fm1','dd','t_max']
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


def ba_multifile_histograms(ba_files, ind_files, indices_names,minmax) : 
    """calculates combined index-oriented and MODIS BA oriented histograms
    
    Computes and returns six histograms using the minmax description 
    provided. Two histograms involve only the indices, which are assumed 
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

    # these two count 0.5 x 0.5 degree cells
    occurrence = ah.AccumulatingHistogramdd(minmax=minmax)
    burned_occurrence = ah.AccumulatingHistogramdd(minmax=minmax)

    # these four count individual modis detections
    burned_forest = ah.AccumulatingHistogramdd(minmax=minmax) 
    burned_not_forest = ah.AccumulatingHistogramdd(minmax=minmax)
    burned_other = ah.AccumulatingHistogramdd(minmax=minmax)
    burned_total = ah.AccumulatingHistogramdd(minmax=minmax)

    ca = trend.CompressedAxes(ind_files[0], 'land') 

    for i_year in range(len(ind_files)) : 
        indfile = ind_files[i_year]
        bafile  = ba_files[i_year]
        count   = bafile.variables['count']
        lc_edges = landcover_classification(bafile.variables['landcover'][:])
        lc_type = rv.CutpointReduceVar(count.shape[:-1], 2, lc_edges)
        timelim = len(indfile.dimensions['days'])-1
        filevars = [ indfile.variables[iname] for iname in indices_names ] 
        for i_day in range(1,timelim) : 
            print i_day
            day_data = [ f[i_day,:] for f in filevars ]
            ba_day = count[...,i_day]
            ba_forest = lc_type.sum(0,ba_day)
            ba_nonforest = lc_type.sum(1,ba_day)
            ba_other     = lc_type.sum(2,ba_day)
            
            ba_forest_cmp = ca.compress(ba_forest)
            ba_nonforest_cmp = ca.compress(ba_nonforest)
            ba_other_cmp = ca.compress(ba_other)
            
            for i_land in range(one_day) : 
                record = ma.array([ data[i_land] for data in day_data] )
                # if any of our coordinates are masked out, skip to next record
                if np.any(record.mask) : 
                    continue
                occurrence.put_record(record)
                burned_weight= 0
                if ba_forest_cmp[i_land] > 0 : 
                    burned_forest.put_record(record, weight=ba_forest_cmp[i_land])
                    burned_weight += ba_forest_cmp[i_land]
                if ba_nonforest_cmp[i_land] > 0 : 
                    burned_not_forest.put_record(record, weight=ba_nonforest_cmp[i_land])
                    burned_weight += ba_nonforest_cmp[i_land]
                if ba_other_cmp[i_land] > 0 : 
                    burned_other.put_record(record, weight=ba_other_cmp[i_land])
                    burned_weight += ba_other_cmp[i_land]
                if burned_weight > 0 : 
                    burned_total.put_record(record, weight=burned_weight) 
                    burned_occurrence.put_record(record)

    return (occurrence, burned_occurrence, burned_forest, burned_not_forest, 
            burned_other, burned_total)

def write_multiyear_histogram_file(outfile, histos, ind_names, minmax)  :
    """write a multiyear histogram netcdf file"""
    ofile = nc.Dataset(outfile, 'w')
        
    # create dimensions
    for i_ind, indname in enumerate(ind_names) : 
        cur_min, cur_max, cur_bins = minmax[i_ind]
        ofile.createDimension(indname, cur_bins)
        cv = ofile.createVariable(indname, np.float64, dimensions=(indname,)) 
        binsize    = float(cur_max - cur_min)/cur_bins
        cv[:] = np.arange(cur_min, cur_max, binsize)
        cv.binsize = binsize

    # store variables
    names = [ 'occurrence', 'burned_occurrence', 'burned_forest', 'burned_not_forest',
                  'burned_other', 'burned_total'] 
    types = [ np.int32, np.int32, np.float64, np.float64, np.float64, np.float64]
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
        bins = [ bins ] * len(years)
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

def select_data(dataframe, names, i_count, indexer, lc_codes=None) : 
    u_lower = indexer.get_unit_val(i_count)
    u_upper = indexer.get_unit_val(np.add(i_count,1))
    
    # initialize to "everything" or "everything in a set of landcover codes"
    if lc_codes is None :
        i_data = np.ones( (dataframe.shape[0],), dtype=np.bool)
    else : 
        i_data = [dataframe.ix[i,1] in lc_codes for i in range(dataframe.shape[0])]
    for low,high,name in zip(u_lower, u_upper, names) :
        i_cur = np.logical_and(dataframe[name]>=low, dataframe[name]<high)
        i_data = np.logical_and(i_data, i_cur)
        
    return dataframe[i_data]

def sparse_multiyear_histogram(years, csv_template, bahistfile, 
                            count_threshold=50, bins=25, out_template=None) : 
    """computes and optionally saves sparse histograms of MODIS BA counts"""
    # open the ba histogram file
    bahist = nc.Dataset(bahistfile)
    counts = bahist.variables['burned_occurrence']
    
    # read all csv files and concatenate
    file_list = []
    for y in years :  
        file_list.append(pd.read_csv(csv_template % y))
    compare = pd.concat(file_list)
    
    # get min/max/bin from multiyear histogram file
    mmb = [] 
    binsizes = [ ]
    for dimname in counts.dimensions: 
        dim = bahist.dimensions[dimname]
        cv = bahist.variables[dimname][:]
        binsizes.append( bahist.variables[dimname].binsize )
        mmb.append( (cv[0], cv[-1]+binsizes[-1], len(dim)))
        
    # create an indexer
    index = ah.init_indexers(mmb)    
    
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
        total = select_data(compare, counts.dimensions, i_bin, index)
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
                         
    
