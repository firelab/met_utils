"""Manipulates burned area

Contains code primarily intended to accumulate a shapefile of burned area
points onto a low-resolution grid.
"""

import qty_index as q
import numpy as np
import astropy.units as u
import astropy.time as time
import aggregator as agg
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
    
    # make indexer for various axes
    lat_idx = q.LinearSamplingFunction(srcvar['nav_lat'][1,0]-srcvar['nav_lat'][0,0],
                                        x_zero=srcvar['nav_lat'][0,0])
    lon_idx = q.LinearSamplingFunction(srcvar['nav_lon'][0,1]-srcvar['nav_lon'][0,0],
                                        x_zero=srcvar['nav_lon'][0,0])    
    time_idx = q.TimeSinceEpochFunction(1*u.day, time.Time('%d:001'%year,format='yday'))
    idx = q.OrthoIndexer([lat_idx, lon_idx, time_idx])
    
    #create the netcdf variable
    count = ba.create_variable('count', np.int, ('y','x','days'))
    ui_count = q.UnitIndexNdArray(count, idx)
    
    # open the shapefile & get layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shapefile, 0) # 0 means read-only. 1 means writeable.
    layer = dataSource.GetLayer()
    
    for feature in layer : 
        loc = feature.GetGeometryRef().GetPoint(0)
        t = feature.GetField("DATE")
        t_obj = time.Time(t)
        
        # increment the correct counter
        ui_count.inc( (loc[1],loc[0],t_obj))
        
    ba.close()
    