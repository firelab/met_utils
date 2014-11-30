"""Manipulates burned area

Contains code primarily intended to accumulate a shapefile of burned area
points onto a low-resolution grid.
"""

import qty_index as q
import numpy as np
import netCDF4 as nc
import astropy.units as u
import astropy.time as time
import aggregator as agg
import dateutil.parser as dup
import pandas as pd
import trend
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
    

def ba_compare_year(indicesfile, bafile, outfile=None) : 
    """collates the various indices with burned area counts"""

    # interesting variables from the indicesfile
    indicesvars = ['gsi_avg','fm1000','fm100']

    indices = nc.Dataset(indicesfile)
    ba = nc.Dataset(bafile)
    count = ba.variables['count']

    ca = trend.CompressedAxes(indices, 'land')

    alldata = []

    for day in range(len(ba.dimensions['days'])) : 
        # compress the count
        day_count = ca.compress(count[...,day])

        # find nonzero counts
        i_nonzero = np.nonzero(day_count)

        # construct dataframe for this day
        todays_data = {"BA Count" : day_count[i_nonzero]}

        for v in indicesvars : 
            day_v = indices.variables[v][day,:]
            todays_data[v] = day_v[i_nonzero]

        alldata.append( pd.DataFrame( todays_data ) ) 

    all_data_frame = pd.concat(alldata, keys=range(len(ba.dimensions['days'])))

    if outfile is not None : 
        all_data_frame.to_csv(outfile)

    return all_data_frame
