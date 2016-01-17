"""Generates indices from CRUNCEP data

Code accepts an ORCHIDEE forcing file and an index filename. It computes various 
indices from the met data and stores them in the outfile.

Code can also summarize indices as percentiles. For this, the user must specify
one or more files containing indices and a summary outfile.
"""

import aggregator as agg
import trend 
import qty_index as q
import astropy.units as u
import astropy.coordinates as c
import astropy.time as t
import numpy as np
import numpy.ma as ma
import fuelmoisture as fm
import met_functions as met
import window as w
import sun
import gsi
import precipitation as p
import percentile as pct
import netCDF4 as nc


DRY_DAY_PRECIP = (3.0 * u.mm).to(u.kg/(u.m**2), equivalencies=p.precipitation())

class ForcingDataset ( agg.NetCDFTemplate ) :
    """Class manages the computation of indices from ORCHIDEE forcing data.
    
    The class is designed to walk through the input file one day's data 
    at a time. Primarily, it allows the caller to register variables of 
    interest which are tracked synchronously.
    """

    def get_forcing(self) : 
        """returns a reference to the forcing dataset"""
        if not hasattr(self, "_nc_forcing") : 
            self._nc_forcing = self._openExemplar()
        return self._nc_forcing
        
    def get_land_axes(self) : 
        """create/cache/return the translator between (x,y) and (land) shapes"""
        if not hasattr(self, "_land_axes") :
            self._land_axes = trend.CompressedAxes(self.get_forcing(), 'land')
        return self._land_axes
        
    def get_xy_indices(self) : 
        if not hasattr(self, "_xy_indices") : 
            ca = self.get_land_axes()
            self._xy_indices = ca.get_grid_indices()
        return self._xy_indices
    
    def get_longitudes(self) : 
        """caches/retrieves longitudes"""
        if not hasattr(self,"_longitudes") :
            forcing = self.get_forcing()
            lon_array = forcing.variables['nav_lon'][:] * u.deg
            self._longitudes = lon_array[self.get_xy_indices()]
        
        return self._longitudes
        
    def get_latitudes(self) : 
        """caches/retrieves latitudes"""
        if not hasattr(self,"_latitudes") : 
            forcing = self.get_forcing()
            lat_array = forcing.variables['nav_lat'][:] * u.deg
            self._latitudes = lat_array[self.get_xy_indices()]
        return self._latitudes

    def get_num_land_points(self) :   
        forcing = self.get_forcing()
        return len(forcing.dimensions['land'])          
        
    def get_daylength_by_lat(self, day) : 
        """Calculates the daylength at each latitude
        
        Daylength depends on the date you're computing it for, rather than 
        being a constant function of location. Make sure "day" is an 
        astropy.time.Time object.
        
        Returns an array of daylengths for each latitude cell in the simulation.
        These correspond to latitude array:
            cruncep.variables['nav_lat'][:,0]
        """
        forcing = self.get_forcing()
        lats = forcing.variables['nav_lat'][:,0] * u.deg
        lons = np.zeros( lats.shape ) 
        earthlocs = c.EarthLocation.from_geodetic(lons,lats)
        
        up = sun.Uptime(sun.SunPosition, day, earthlocs)
        return up.approximate_daylength()
        
    def get_daylength_lookup(self, day) : 
        forcing = self.get_forcing()
        lats = forcing.variables['nav_lat'][:,0] * u.deg
        values = self.get_daylength_by_lat(day).to(u.hour)
        return q.LookupTable(lats[0], (lats[1]-lats[0]), values)
        
    def get_timestep(self) :
        """caches/calculates the timestep from the info in the file""" 
        if not hasattr(self, "_timestep") : 
            forcing = self.get_forcing()
            times = forcing.variables['time'][:2] * u.s
            self._timestep = (times[1] - times[0])
        return self._timestep
        
    def get_samples_per_day(self) : 
        if not hasattr(self, '_spd') :
            timestep = self.get_timestep()
            self._spd = ((24*u.hour)/timestep).to(u.dimensionless_unscaled).astype(np.int)
        return self._spd    
        
    def get_buffer_group(self) : 
        if not hasattr(self, "_buffers") : 
            self._buffers = q.DLTSGroup() 
        return self._buffers
        
    def get_derived_buffers(self) : 
        if not hasattr(self, "_derived") : 
            self._derived = q.DLTSGroup()
            self._derived.template = self._buffers.template
        return self._derived
        
    def register_variable(self, varname, unit) :
        """adds "varname" to the list of variables to track""" 
        bufs = self.get_buffer_group()
        
        forcing = self.get_forcing()
        ncvar = forcing.variables[varname]
        if not bufs.template_ready() : 
            time_axis = ncvar.dimensions.index('tstep')
            bufs.add(varname, q.DiurnalLocalTimeStatistics(
                ncvar, time_axis,
                self.get_timestep(), self.get_longitudes(), unit=unit))
        else : 
            bufs.create(varname, ncvar, unit)
                
        return bufs.get(varname)
        
    def compute_rh(self) : 
        """Compute RH for the current utc day
        
        This calls "next()" on the internal RH object. No need to manage it."""
        bufs = self.get_buffer_group()
        derived = self.get_derived_buffers()
        if "rh" in derived.group : 
            # one day big
            p = bufs.get('PSurf').get_utc_day()
            q = bufs.get('Qair').get_utc_day()
            t = bufs.get('Tair').get_utc_day()
            rh = derived.get('rh')
            rh.source = met.calc_rh_spec_humidity(q,p,t)
            rh.next()
        else :
            # 2 days big
            p = bufs.get('PSurf').get_buffer()
            q = bufs.get('Qair').get_buffer()
            t = bufs.get('Tair').get_buffer()
            rh = derived.create_computed("rh",met.calc_rh_spec_humidity(q,p,t), 
                    u.pct)
            
        return rh
        
    def compute_afternoon_vpd(self) :   
        bufs = self.get_buffer_group()
        t = bufs.get('Tair').ref_val()
        q = bufs.get('Qair').ref_val()
        p = bufs.get('PSurf').ref_val()
        vp_act = met.calc_vp_spec_humidity(q,p)
        return met.calc_vpd(t,vp_act)
        
    def compute_precip_duration(self) : 
        bufs = self.get_buffer_group()
        
        rainf = bufs.get('Rainf')
        landpts = self.get_num_land_points()
        
        # count the number of times rain was observed
        # in the past day
        rain_count = np.empty( (landpts,) )
        raining = (rainf.get_preceeding_day() > 0)
        i_raining = [ slice(None,None,None) ] * 2
        for i in range(landpts) : 
            i_raining[not rainf.time_axis] = i
            rain_count[i] = np.count_nonzero(raining[i_raining])
        
        daily_obs = self.get_samples_per_day() / u.day
        return fm.precip_duration_sub_day(rain_count, daily_obs)

    def compute_eqmc_bar(self, daylengths) : 
        bufs = self.get_buffer_group()
        derived = self.get_derived_buffers()
        
        temps = bufs.get('Tair')
        rh    = derived.get('rh')
        
        t_min = temps.min()
        t_max = temps.max()
        rh_min = rh.min()
        rh_max = rh.max()
        
        return fm.eqmc_bar(daylengths, t_max, t_min, rh_max, rh_min)
        
       

    def next(self) :
        """advance all of the registered variables to the next day"""
        bufs = self.get_buffer_group()
        bufs.next()
                        
    def close(self) :
        """closes the input and output netCDF files""" 
        if hasattr(self, "_nc_forcing") : 
            self._nc_forcing.close()
            
        super(ForcingDataset,self).close()
        
def indices_year(y, forcing_template, out_template) : 
    """Processes one year.
    
    Runs the calculation of indices for one year given an orchidee forcing
    file for that year."""
    
    # figure out filenames and open dataset.
    forcing_file = forcing_template % (y,)
    out_file = out_template % (y,)
    ds = ForcingDataset(forcing_file, out_file)
    
    # register required variables from NetCDF file so they are tracked
    qair = ds.register_variable("Qair", u.dimensionless_unscaled)
    tair = ds.register_variable("Tair", u.Kelvin)
    ds.register_variable("PSurf", u.Pa)
    rainf = ds.register_variable("Rainf", u.kg / (u.m**2) / u.s )
    swdown = ds.register_variable("SWdown", u.W / (u.m**2))
    
    print "NetCDF variables registered"
    # setup netcdf dimensions in output file
    ds.copyDimension("land")
    ds.copyDimension("y")
    ds.copyDimension("tstep")
    ds.createDimension("days", 365)
    
    # copy basic data
    ds.copyVariable('nav_lat')
    ds.copyVariable('nav_lon')

    
    print "Variables copied from forcing file"
    
    # create netcdf variables to hold indices in output file
    daylength = ds.create_variable("daylength", ("days","y"), np.float32)
    daylength.title = "Day Length"
    daylength.units = "hours"
    
    rh = ds.create_variable("rh", ("tstep","land"), np.float32)
    rh.long_name = "Relative Humidity"
    rh.units = "fraction"
    
    rh_max = ds.create_variable("rh_max", ("days","land"), np.float32)
    rh_max.long_name = "Maximum RH" 
    rh_max.units = "fraction"
    
    rh_min = ds.create_variable("rh_min", ("days","land"), np.float32)
    rh_min.long_name = "Minimum RH"
    rh_min.units = "fraction"
    
    rh_afternoon = ds.create_variable("rh_afternoon", ('days', 'land'),np.float32)
    rh_afternoon.long_name = "RH in the afternoon"
    rh_afternoon.units = "fraction"
    
    t_min = ds.create_variable("t_min", ('days','land'), np.float32)
    t_min.long_name = "minimum temperature for the 24 hours preceeding burning period"
    t_min.units = "K"
    
    t_max = ds.create_variable("t_max", ('days', 'land'), np.float32)
    t_max.long_name = "maximum temperature for the 24 hours preceeding burning period"
    t_max.units = "K"
    
    t_afternoon = ds.create_variable("t_afternoon", ('days','land'), np.float32)
    t_afternoon.long_name = "mid-burning-period temperature"
    t_afternoon.units = "K"
    
    i_tmin = ds.create_variable("i_tmin", ('days','land'), np.float32)
    i_tmin.long_name = "GSI minimum temperature index component"
    i_tmin.units = 'dimensionless'
    
    i_photo = ds.create_variable('i_photo', ('days','land'), np.float32)
    i_photo.long_name = 'GSI photoperiod index component'
    i_photo.units = 'dimensionless'
    
    i_vpd = ds.create_variable('i_vpd', ('days','land'), np.float32)
    i_vpd.long_name = 'GSI vapor pressure deficit index component'
    i_vpd.units = 'dimensionless'
    
    i_gsi = ds.create_variable('gsi', ('days','land'), np.float32)
    i_gsi.long_name = 'GSI Index'
    i_gsi.units = 'dimensionless'

    gsi_days = 10
    i_gsi_avg = ds.create_variable('gsi_avg', ('days','land'), np.float32)
    i_gsi_avg.long_name = 'GSI Index (%d day running average)' % gsi_days
    i_gsi_avg.units = 'dimensionless'
    
    precip = ds.create_variable('precip_duration', ('days','land'), np.float32)
    precip.long_name = "Duration of precipitation"
    precip.units = 'hours'
    
    tot_precip = ds.create_variable('total_precip', ('days','land'), np.float32)
    tot_precip.long_name = 'total precipitation over last 24 hours'
    tot_precip.units = 'kg / m^2'
    
    dd_days = 30
    dd = ds.create_variable('dd', ('days','land'), np.int)
    dd.long_name = 'dry day weighting (%d day window)' % dd_days
    dd.units = 'none'
    
    eqmc_bar = ds.create_variable('eqmc_bar', ('days','land'), np.float32)
    eqmc_bar.long_name = 'Weighted daily average equilibrium moisture content'
    eqmc_bar.units = 'percent'
    
    fm1000 = ds.create_variable('fm1000', ('days','land'), np.float32)
    fm1000.long_name = '1000 hour fuel moisture'
    fm1000.units = 'percent'
    
    fm100  = ds.create_variable('fm100', ('days','land'), np.float32)
    fm100.long_name = '100 hour fuel moisture'
    fm100.units = 'percent'
    
    fm10 = ds.create_variable('fm10', ('days','land'), np.float32)
    fm10.long_name = '10 hour fuel moisture'
    fm10.units = 'percent'
    
    fm1 = ds.create_variable('fm1', ('days','land'), np.float32)
    fm1.long_name = '1 hour fuel moisture'
    fm1.units = 'percent'
    
    print "Output NetCDF variables created"
    
    
    # moving window of GSI data
    gsi_window = w.MovingWindow(i_gsi.shape, 0, gsi_days)
    
    # moving window to compute dry day weights
    dd_window = w.SequenceWindow(dd.shape, 0, dd_days)
    
    # initialize fuel moisture calculators
    fm1000_calc = fm.ThousandHourFM(fm1000.shape, 0)
    fm100_calc  = fm.HundredHourFM(fm100.shape, 0)
        
    # base time
    time_start = t.Time('%04d:001'%y, format='yday', scale='ut1')
    if ( (y < 1960) or (y > 2013) ) :
        time_start.delta_ut1_utc=0
    
    # loop over days
    # the cur_day property is the index of the next day to read in from the file.
    # since the netcdf file is 0-based, indices are [0, 365)
    while qair.cur_day < 365 : 
        i_day = qair.cur_day - 1 
        print qair.cur_day
        
        # calculate daylengths, store as hours
        day = time_start + (i_day*u.day)
        if (hasattr(time_start,"delta_ut1_utc")) : 
            day.delta_ut1_utc = time_start.delta_ut1_utc
            
        daylength_lut = ds.get_daylength_lookup(day)
        daylength[i_day,:] = daylength_lut.values
        
        # pull out/store minimum, maximum and afternoon temps
        t_min[i_day,:] = tair.min(unitted=False)
        t_max[i_day,:] = tair.max(unitted=False)
        t_afternoon[i_day,:] = tair.ref_val(unitted=False)
        
        # calculate RH & store
        rh_dlts = ds.compute_rh()
        rh_dlts.store_day(rh)
        rh_max[i_day,:] = rh_dlts.max(unitted=False)
        rh_min[i_day,:] = rh_dlts.min(unitted=False)
        rh_afternoon[i_day,:] = rh_dlts.ref_val(unitted=False)
        
        # calculate GSI indices and store
        i_tmin[i_day,:] = gsi.calc_i_tmin(tair.min())
        cell_daylengths = daylength_lut.get(ds.get_latitudes())
        i_photo[i_day,:] = gsi.calc_i_photo(cell_daylengths)
        i_vpd[i_day,:] = gsi.calc_i_vpd(ds.compute_afternoon_vpd())
        gsi_vals = i_tmin[i_day,:] * i_photo[i_day,:] * i_vpd[i_day,:]
        i_gsi[i_day,:] = gsi_vals

        # calculate running GSI average and store
        gsi_window.put(gsi_vals)
        if gsi_window.ready() : 
            i_gsi_avg[i_day,:] = gsi_window.mean()
            
        # calculate precipitation duration and store
        precip_val = ds.compute_precip_duration()
        precip[i_day,:] = precip_val
        
        # calculate total precip over last 24 hours, and dry day flag
        tot_precip_val = (rainf.sum() * ds.get_timestep()).to(u.kg/(u.m**2))
        tot_precip[i_day,:] = tot_precip_val
        dd_window.put( tot_precip_val < DRY_DAY_PRECIP )
        dd[i_day,:] = dd_window.all_runs()
        
        # calculate daily avg equilibrium moisture content
        eqmc_bar_val = ds.compute_eqmc_bar(cell_daylengths)
        eqmc_bar[i_day,:] = eqmc_bar_val
        
        # calculate fuel moisture content
        fm1000[i_day,:] = fm1000_calc.compute(eqmc_bar_val, precip_val).to(u.percent)
        fm100_today  = fm100_calc.compute(eqmc_bar_val, precip_val)
        fm100[i_day,:] = fm100_today.to(u.percent)
        fm1_today, fm10_today = fm.oneten_ofdm(
                t_afternoon[i_day,:]*u.K, 
                rh_afternoon[i_day,:]*u.dimensionless_unscaled,
                swdown.ref_val(), fm100_today)
        fm10[i_day,:] = fm10_today.to(u.percent)
        fm1[i_day,:] = fm1_today.to(u.percent)              
                
        
        # first time through, store the first day's data
        if i_day == 1 : 
            rh_dlts.store_day(rh, current=False)
        
        # load next day for all variables
        ds.next()
    
    ds.close()
    
def multifile_open(datasets, years) : 
    if years is not None : 
        files = [ ]
        for y in years : 
            files.append(nc.Dataset(datasets %y))
        datasets = files
    return datasets
    

def multifile_minmax(datasets, indices, years=None) : 
    """calculates minimum and maximum of indices across multiple files
    
    You may call this function with a list of previously opened NetCDF 
    datasets, or you may provide a template for the filename and a list of 
    years. If you provide a template, this function will open and close
    the files for you.
    """
    
    datasets = multifile_open(datasets, years)
    
    num_ind = len(indices)
    minvals = ma.masked_all( (num_ind,), dtype=np.float64)
    maxvals = ma.masked_all( (num_ind,), dtype=np.float64)
    for i_year in range(len(datasets)) : 
        indfile = datasets[i_year]
        timelim = len(indfile.dimensions['days'])-1
        for i_indices in range(num_ind) : 
            index_vals = indfile.variables[indices[i_indices]][1:timelim,:]
            cur_max = np.max(index_vals)
            cur_min = np.min(index_vals)
            if minvals[i_indices] is ma.masked : 
                minvals[i_indices] = cur_min
                maxvals[i_indices] = cur_max
            else : 
                minvals[i_indices] = min(cur_min, minvals[i_indices])
                maxvals[i_indices] = max(cur_max, maxvals[i_indices])

    if years is not None : 
        for f in minmax_indfiles : 
            f.close()

    return (minvals,maxvals)

def percentile_indices(datasets, indices, outfile, years=None, 
    land_dim='land', time_slice=None) :
    """Calculates cutpoints for integer percentile classes for the named indices.
    
    The indices for which percentiles are desired should be named in the 
    indices variable.
    
    Input files may be specified either by providing an array of previously 
    opened datasets in the "dataset" parameter, or by providing a filename 
    template in the dataset parameter and a list of years to be included.
    
    The output file will contain a variable for each named index which will have
    an array of cutpoints for integer percentiles for each land pixel. 
    
    All of the indices you specify need to have an identical dimensionality in
    the input datasets. Cannot mix and match.
    
    You may change the default value of the name of the "land" dimension using 
    the land_dim" parameter. If you would like to include a subset of data
    each year along the time dimension, provide the slicing expression as 
    "time_slice". 
    """
    datasets = multifile_open(datasets, years)
    
    out_templ = agg.NetCDFTemplate(datasets[0].filepath(), outfile)
    
    num_ind = len(indices) 
    num_years = len(datasets)
    num_land = len(datasets[0].dimensions[land_dim])
    d = datasets[0]
    v = d.variables[indices[0]]
    ipos_land = v.dimensions.index(land_dim)
    ipos_time = not (ipos_land)
    
    # pre-compute the number of time elements which need to be assembled 
    # from the files before computing percentiles.
    # Pre-computing this makes the assumption that every year and every
    # index has the same time dimension.
    if time_slice is not None : 
        num_time = (time_slice.stop - time_slice.start) + 1
        num_time *= num_years
        out_templ._ncfile.start_day = time_slice.start
        out_templ._ncfile.end_day   = time_slice.stop-1
    else :
        one_year = d.dimensions[v.dimensions[ipos_time]]
        num_time = one_year * num_years
        time_slice = slice(None, None, None)
        out_templ._ncfile.start_day = 0
        out_templ._ncfile.end_day = one_year - 1
        
    # prep the output netcdf file
    out_templ.copyVariable('nav_lat')
    out_templ.copyVariable('nav_lon')
    out_templ.createDimension('percentile_cutpoints', 101)

    
    # loop over all the indices we're collecting data for
    for i_indices in range(num_ind) : 
        
        # create a variable in the output file to contain the percentiles
        ind_name = indices[i_indices]
        cur_dtype = d.variables[ind_name].dtype
        out_v = out_templ.create_variable(ind_name, 
                   ('land','percentile_cutpoints'), cur_dtype)
        
        # loop over all the land pixels
        for i_land in range(num_land) :
            if (i_land % 100) == 0 : 
                print "%s - Pixel %d" % (ind_name, i_land)
              
            pf = pct.PercentileFactory(num_time)
            
            # collect the data for a single land pixel from all the input
            # files
            for i_year in range(num_years) : 
                d = datasets[i_year]
                v = d.variables[ind_name]
                if ipos_land == 0 : 
                    pf.add_data(v[i_land, time_slice])
                else : 
                    pf.add_data(v[time_slice, i_land])
                
            samp_func = pf.compute_percentile()
            
            out_v[i_land,:] = samp_func.cutpoints
            
    out_templ.close()

def apply_percentile_year(dataset, pctfile, outfile, land_dim='land',
                     time_slice=None): 
    """Computes indices as percentile values and stores in output file.
    
    The indices which are converted to percentile values are present as variables
    in the pctfile. pctfile contains the cutpoints for each variable's percentile
    bins. The output is an integer array which is effectively the result of 
    the "ceil" function. Valid percentile values are 1 to 100, inclusive.
    Index values between the minimum and the 1st percentile cutpoint (inclusive) 
    get value 1, between the 1st (exclusive) and 2nd (inclusive) cutpoint get value
    2, and so on.
    
    Input file is specified by providing a filename as a dataset.
    
    Outfile specifies a name for the output file. The output file will contain 
    a variable for each index, where the indices are represented as percentiles.
    """
    ds = nc.Dataset(dataset)
    out_templ = agg.NetCDFTemplate(dataset, outfile)
    
    pct_ds = nc.Dataset(pctfile)
    indices = [v for v in pct_ds.variables.keys() if v not in pct_ds.dimensions.keys()] 
    indices.remove('nav_lat')
    indices.remove('nav_lon')
    num_land = len(ds.dimensions[land_dim])

    if time_slice is not None : 
        out_templ._ncfile.start_day = time_slice.start
        out_templ._ncfile.end_day   = time_slice.stop - 1

    # copy geolocation
    out_templ.copyVariable('nav_lat')
    out_templ.copyVariable('nav_lon')
    
    # pre-compute the number of time elements which need to be assembled 
    # from the files before computing percentiles.
    # Pre-computing this makes the assumption that every year and every
    # index has the same time dimension.
    if time_slice is None : 
        time_slice = slice(0, len(ds.dimensions['days']), None)

    # loop over indices
    for ind in indices : 
        print ind
        in_index = ds.variables[ind]
        ipos_land = in_index.dimensions.index(land_dim)
        pct_index = pct_ds.variables[ind][:]
        
        out_v = out_templ.create_variable(ind,
            in_index.dimensions, np.int8, fill=-127)
        
                
        #loop over time slice 
        for day in range(time_slice.start,time_slice.stop) : 
            print day
            
            ceil_pct = ma.masked_all( (num_land,), dtype=np.int8)
            if ipos_land == 0 : 
                in_day = in_index[:,day]
                
                
                # searchsorted with side = 'right' is essentially the 
                # ceil function.
                for pix in range(num_land) : 
                    if not ma.is_masked(in_day[pix]) :
                        ceil_pct[pix] = np.searchsorted(pct_index[pix,:], in_day[pix])

                # only thing with value 0 is the minimum index value.
                # wrap this into the first percentile bin, essentially 
                # making the first bin a closed interval on both sides.
                ceil_pct[(ceil_pct==0) & (~ceil_pct.mask)] =1 
                
                out_v[:, day] = ceil_pct
            else : 
                in_day = in_index[day,:]
                for pix in range(num_land) : 
                    if not ma.is_masked(in_day[pix]): 
                        ceil_pct[pix] = np.searchsorted(pct_index[pix,:], in_day[pix])

                ceil_pct[(ceil_pct==0) & (~ceil_pct.mask)]=1
                out_v[day, :] = ceil_pct
    ds.close()
    out_templ.close()
    pct_ds.close()

class IndexManager (object) : 
    """Encapasulates day-by-day access to a set of files of indices.
    
    The primary function of this object is to compile vectors of 
    indices from the parallel arrays found in the file of indices. Each
    vector represents a single cell, and an entire day is processed at once.
    
    An optional geographic mask is accepted.
    """
    def __init__(self, indices_names, geog_mask=None) :
        """Initializes a manager object.
        
         ind_series is a time series of files of indices.
         indices_names is a set of indices to read from the file.
         geog_mask is a 1D array of booleans where true values indicate that
          the cell is within the ROI.
        """ 
        self.indices_names   = indices_names
        self.geog_mask  = geog_mask
        
    def set_mask(self, geog_mask) : 
        self.geog_mask = geog_mask
        
    
    def get_indices_vector(self, ind_ds, i_day) :
        """compiles a vector of indices for each grid cell on the specified day.
        
        This is used when the specific dataset and day index are known.
        The 1D vector of valid land pixels is returned, along with a 2D
        (land, index) array of index values. To obtain only those index
        values over valid land pixels, you must select by the returned
        1D vector."""
        
        one_day = len(ind_ds.dimensions['land'])
        # get variable references for each index
        filevars = [ ind_ds.variables[iname] 
                            for iname in self.indices_names ] 

        # assemble the desired indices into parallel arrays
        #day_data = [ f[i_day,:] for f in filevars ]

        # construct an array of records for all land pixels in a single 
        # day. Also construct a mask which can be used to pull data from
        # the weights arrays
        records = ma.zeros( (one_day, len(filevars)))
        
        # compile all the records for a single day (column-wise)
        for i_data in range(len(filevars)):
            records[:,i_data] = filevars[i_data][i_day,:]
            
        # filter out pixels where any of the indices are missing. (row-wise)    
        # Merge in the geographic filter.
        if len(filevars) > 1 : 
            land_data = np.any(records.mask, axis=1)
        else : 
            land_data = records.mask.squeeze()
        
        # Merge in the geog_mask, if specified    
        if self.geog_mask is None :
            land_data = np.logical_not(land_data)
        else : 
            land_data = self.geog_mask & np.logical_not(land_data)
                        
        return (land_data, records)
        
    def get_day(self, ind_series, day) : 
        """Looks up the file and position within file given the time-series and date"""
        ind_ds, i_day = ind_series.get_location(day)
        return self.get_indices_vector(ind_ds, i_day)
