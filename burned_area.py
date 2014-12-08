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
import trend
from osgeo import ogr


FOREST_LC = range(1,6)

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
    
    time_samples = len(ba.dimensions['days'])  
    if reduction is not None : 
        grid_reducer = rv.ReduceVar(count.shape, 3, reduction)
        cmp_reducer  = rv.ReduceVar(indicesvars[0].shape, 0, reduction)
        time_samples = grid_reducer.reduced

    ca = trend.CompressedAxes(indices, 'land')

    alldata = []
    days = [] 

    for i_time in range(time_samples) : 
        day_data = [] 
        active_lc = [] 
        

        if reduction is None :
            count_slice = count[...,i_time]
        else :
            count_slice = grid_reducer.sum(i_time, count)  
            
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
