"""Performs fore-or-hindcasting based on percentile index values"""

import netCDF4 as nc
import trend
import numpy as np
import time_series as ts
import orchidee_indices as oi
import accum_hist as ah
from collections import namedtuple

lc_type_names = ['forest','not_forest', 'other', 'total']

Landcover = namedtuple("Landcover", lc_type_names+['bin_centers', 'bin_edges'])

def get_ratios(pcthisto) : 
    """calculates the ratio of measured burning activity to measured index value occurrence"""
    pctfile = nc.Dataset(pcthisto) 
    
    burn_forest = pctfile.variables['ratio_burned_forest'][:]
    burn_not_forest =  pctfile.variables['ratio_burned_not_forest'][:]
    burn_other = pctfile.variables['ratio_burned_other'][:]
    burn_total = pctfile.variables['ratio_burned_total'][:]
    
    bin_dim = pctfile.variables['ratio_burned_forest'].dimensions[0]
    bin_left = pctfile.variables[bin_dim][:]
    binsize = pctfile.variables[bin_dim].binsize
    n_bins = bin_left.shape[0]
    
    bin_edges = np.empty( (n_bins+1,), dtype=bin_left.dtype)
    bin_edges[:-1] = bin_left
    bin_edges[-1] = bin_left[-1] + binsize
    
    bin_centers = bin_left + (binsize/2)
    
    pctfile.close()
    
    return Landcover(forest=burn_forest,
                   not_forest=burn_not_forest,
                   other=burn_other,
                   total=burn_total,
                   bin_centers=bin_centers,
                   bin_edges=bin_edges) 

def calc_mask(series, periods, pctfile, land_dim='land') :
    # read the bounding box if recorded in the file. 
    pct_ds = nc.Dataset(pctfile)
    if not ('max_lat' in pct_ds.ncattrs()) :
        pct_ds.close()
        return None
    max_lat = pct_ds.max_lat
    min_lat = pct_ds.min_lat
    max_lon = pct_ds.max_lon
    min_lon = pct_ds.min_lon
    
    pct_ds.close()
    
    ds = series.get_dataset(periods.first())
    ca = trend.CompressedAxes(ds, land_dim)
    
    nav_lat = ds.variables['nav_lat'][:]
    nav_lon = ds.variables['nav_lon'][:]
    
    mask_2d = (nav_lat >= min_lat) & \
              (nav_lat <= max_lat) & \
              (nav_lon >= min_lon) & \
              (nav_lon <= max_lon)
    mask_1d = ca.compress(mask_2d)
    
    geog_box = (min_lon, max_lon, min_lat, max_lat)
    
    return (mask_1d, geog_box)
            
def calc_occurrence(p_date, index_series, index_manager,
                    model_periods, bin_edges) : 
    """Calculate the occurrence histogram for a prediction interval"""                

    # fetch the handle to the correct index file
    ds = index_series.get_dataset(p_date)
    
    occ = ah.AccumulatingHistogramdd([ (np.min(bin_edges),
                                              np.max(bin_edges),
                                              len(bin_edges)-1) ])

    # build a histogram of the index occurrence, one day at a time 
    for i_day in model_periods.interval() : 
        valid, x = index_manager.get_indices_vector(ds, i_day)
        occ.put_batch(x[valid])
        
    return occ
            
def create_cast_file(filename, periods, bin_centers, 
                      time_slice=None, geog_box=None) : 
    """Creates and initializes the fore/hindcast netCDF file."""
    
    ofile = nc.Dataset(filename, 'w')
    
    if time_slice is not None : 
        ofile.start_day = time_slice.start
        ofile.end_day   = time_slice.stop - 1
        
    # record the geographic bounding box used to generate this data
    if geog_box is not None : 
        ofile.min_lon = geog_box[0]
        ofile.max_lon = geog_box[1]
        ofile.min_lat = geog_box[2]
        ofile.max_lat = geog_box[3]

    # create the dimensions
    ofile.createDimension('period', periods.length())
    ofile.createDimension('landcover', len(lc_type_names))
    ofile.createDimension('percentile', len(bin_centers))
    ofile.createDimension('lc_name_len', 20)
    
    # create coordinate variables
    pct = ofile.createVariable('percentile',  np.float, ('percentile',))    
    pct[:] = bin_centers
    pct.binsize = bin_centers[1] - bin_centers[0]
    
    lc = ofile.createVariable('landcover', 'c', ('landcover','lc_name_len'))
    for i in range(len(lc_type_names)) :
        n = lc_type_names[i]
        lc[i,:len(n)] = n
        
    period = ofile.createVariable('period', np.int, ('period',))
    epoch = periods.first()
    period.units = 'days since {:%Y-%m-%d}'.format(epoch)
    period[:] = [ (p-epoch).days for p in periods.interval() ] 
    
    # create variable to store annual occurrence histograms
    occ = ofile.createVariable('occurrence', np.int, ('period','percentile'))
    occ.units = 'counts'
    occ.long_name = 'Occurrence histograms for the index within the domain'
    
    ba = ofile.createVariable('burned_area_histograms', np.float, 
           ('period', 'landcover','percentile'))
    ba.units = 'MODIS pixel counts'
    ba.long_name = 'Prediction of deduplicated MODIS burned area detections per landcover type and percentile bin.'
    
    bat = ofile.createVariable('burned_area_totals', np.float,
              ('period', 'landcover'))
    bat.units = 'MODIS pixel counts'
    bat.long_name = 'Prediction of deduplicated MODIS burned area detections per landcover type'    
    
    return ofile

def cast(index_series, indices, cast_periods, 
         model_periods, pcthisto, outfile) : 
    """
    Perform the fore-or-hind cast.
    
    This function expects to take a time series of met index predictions and a 
    set of known relationships to burned area over a small set of landcover types
    in order to produce burned area predictions. This function automatically
    applies the same geographic bounds 
    
    Within each fore-or-hind cast interval: 
    
    The met index values for the entire spatio-temporal domain of the prediction
    period are histogrammed. This histogram is then multiplied element-for-element
    with the burned area to index occurrence ratio for each of the landcover types, 
    producing estimates for the amount of burned area at each index value. 
    Finally, the burned area histograms for each landcover type are summed over 
    the histogram bins to produce a single prediction for burned area for each 
    landcover type.
    
    index_series: time series of percentile index data
    indices: list of index names
    cast_periods: iterator which advances in units of one forecast period
    model_periods: iterator which returns each sub-forecast period
       slice of data (usually days).
    pcthisto: netcdf file containing histograms of burning activity binned by 
       percentile index
    outfile: file to hold the fore/hind-cast
    """
    
    known_ratios = get_ratios(pcthisto)
    mask, geog_box = calc_mask(index_series, cast_periods, pcthisto)
    manager = oi.IndexManager(indices, mask)
    ofile = create_cast_file(outfile, cast_periods, known_ratios.bin_centers,
                geog_box=geog_box, time_slice=model_periods)
    
    # loop over the fore/hind cast periods (typically years)
    i_period = 0
    out_occ = ofile.variables['occurrence']
    out_ba_hist = ofile.variables['burned_area_histograms']
    out_ba  = ofile.variables['burned_area_totals']
    for p in cast_periods.interval() : 
        
        # calculate and store the cumulative occurrence histogram 
        # for this prediction interval
        occurrence = calc_occurrence(p, index_series, manager,  
                                  model_periods,
                                  known_ratios.bin_edges)
        out_occ[i_period,:] = occurrence.H  
                                
        for i_lc in range(len(lc_type_names)) : 
            lc = lc_type_names[i_lc]
            ba_hist = getattr(known_ratios,lc) * occurrence.H
            ba = ba_hist.sum()
            
            out_ba_hist[i_period,i_lc,:] = ba_hist
            out_ba[i_period,i_lc] = ba
            
        i_period += 1
        
    ofile.close()  
    
def do_cast(index_pattern, indices, years, pcthisto, outfile, 
            time_dim='days', time_slice=None) :
    """Creates the series objects and iterators, then performs the fore-or-hind cast.
    
     index_pattern : filename pattern which should expect to receive a year
     indices       : a list of indices to process in this run
     years         : a slice object with the start and stop years
     pcthisto      : percentile histogram file containing ratios of BA to occurrence
     outfile       : name of output file
     time_dim      : name of the time dimension in the index files
     time_slice    : day-of-year range to include in processing 
     """
    
    # how to translate a date to the correct file to open.        
    idx_series = ts.TimeSeries(index_pattern, time_dim) 
    
    # the years to loop over
    cast_periods = ts.AnnualInterval(years.start, years.stop)
    
    # subset of days within year to loop over
    if time_slice is None : 
        model_periods = ts.IntegerInterval(0,365)
    else : 
        model_periods = ts.IntegerInterval(time_slice.start, time_slice.stop)
        
    cast(idx_series, indices, cast_periods, model_periods, pcthisto, outfile)
    
    
