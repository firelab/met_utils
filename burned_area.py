"""Manipulates burned area

Contains code primarily intended to accumulate a shapefile of burned area
points onto a low-resolution grid.
"""

import qty_index as q
import numpy as np
import astropy.units as u
import astropy.time as time
import aggregator as agg
import dateutil.parser as dup
from osgeo import ogr

def ba_year(year, template, ncfile, shapefile)  :
    """creates a burned area netcdf file from the template
    
    The template provided should be a netcdf file of indices"""
    
    # open file / copy info from template
    ba = agg.NetCDFTemplate(template, ncfile)
    ba.copyVariable('nav_lat')
    ba.copyVariable('nav_lon')
    ba.copyDimension('days')
    srcvar = ba._ncfile.variables
    dims   = ba._ncfile.dimensions
    
    # make indexer for various axes
    delta_lat=  (srcvar['nav_lat'][1,0]-srcvar['nav_lat'][0,0])*u.deg
    delta_lon=  (srcvar['nav_lon'][0,1]-srcvar['nav_lon'][0,0])*u.deg
    lat_idx = q.LinearSamplingFunction(1./delta_lat, x_zero=srcvar['nav_lat'][0,0]*u.deg)
    lon_idx = q.LinearSamplingFunction(1./delta_lon, x_zero=srcvar['nav_lon'][0,0]*u.deg)    
    time_idx = q.TimeSinceEpochFunction(1*u.day, time.Time('%d:001'%year,format='yday'))
    idx = q.OrthoIndexer([lat_idx, lon_idx, time_idx])
    
    #create the netcdf variable
    count = ba.create_variable('count', ('y','x','days'), np.int)
    cache = np.zeros( (len(dims['y']),len(dims['x']), len(dims['days'])), dtype=np.int32)
    ui_count = q.UnitIndexNdArray(cache, idx)
    
    # open the shapefile & get layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shapefile, 0) # 0 means read-only. 1 means writeable.
    layer = dataSource.GetLayer()
    
    counter = 0
    for feature in layer : 
        if (counter % 100) == 0 : 
            print counter
        counter += 1
        geom = feature.GetGeometryRef()
        if geom.GetGeometryName() == 'POLYGON' : 
            loc = geom.GetGeometryRef(0).GetPoint(0) * u.deg
            t = feature.GetField("DATE")
            dt = dup.parse(t)
            # convert to UTC if necessary
            if dt.tzinfo is not None: 
                dt = (dt - dt.utcoffset()).replace(tzinfo=None)
                if dt.timetuple()[7] >= 366 : 
                    continue
            t_obj = time.Time(dt)
        
            # increment the correct counter
            ui_count.inc( (loc[1],loc[0],t_obj))
        
    count[:] = cache[:]
    ba.close()
    
