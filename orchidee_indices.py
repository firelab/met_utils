"""Generates indices from CRUNCEP data

Code accepts an ORCHIDEE forcing file and an index filename. It computes various 
indices from the met data and stores them in the outfile.
"""

import aggregator as agg
import trend 
import qty_index as q
import astropy.units as u
import astropy.coordinates as c
import astropy.time as t
import numpy as np
import fuelmoisture as fm
import met_functions as met
import sun
import gsi

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
        values = self.get_daylength_by_lat(day)
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
#    rainf = ds.register_variable("Rainf")
#    swdown = ds.register_variable("SWdown")
    
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
    rh.units = "percent"
    
    rh_max = ds.create_variable("rh_max", ("days","land"), np.float32)
    rh_max.long_name = "Maximum RH" 
    rh_max.units = "percent"
    
    rh_min = ds.create_variable("rh_min", ("days","land"), np.float32)
    rh_min.long_name = "Minimum RH"
    rh_min.units = "percent"
    
    rh_afternoon = ds.create_variable("rh_afternoon", ('days', 'land'),np.float32)
    rh_afternoon.long_name = "RH in the afternoon"
    rh_afternoon.units = "percent"
    
    t_min = ds.create_variable("t_min", ('days','land'), np.float32)
    t_min.long_name = "minimum temperature for the 24 hours preceeding burning period"
    t_min.units = "K"
    
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
    
    print "Output NetCDF variables created"
        
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
        daylength[i_day,:] = daylength_lut.values.to(u.hour)
        
        # pull out/store minimim and afternoon temps
        t_min[i_day,:] = tair.min(unitted=False)
        t_afternoon[i_day,:] = tair.ref_val(unitted=False)
        
        # calculate RH & store
        rh_dlts = ds.compute_rh()
        rh_dlts.store_day(rh)
        rh_max[i_day,:] = rh_dlts.max(unitted=False)
        rh_min[i_day,:] = rh_dlts.min(unitted=False)
        rh_afternoon[i_day,:] = rh_dlts.ref_val(unitted=False)
        
        # calculate GSI indices and store
        i_tmin[i_day,:] = gsi.calc_i_tmin(tair.min())
        i_photo[i_day,:] = gsi.calc_i_photo(daylength_lut.get(ds.get_latitudes()))
        i_vpd[i_day,:] = gsi.calc_i_vpd(ds.compute_afternoon_vpd())
        i_gsi[i_day,:] = i_tmin[i_day,:] * i_photo[i_day,:] * i_vpd[i_day,:]
        
        # first time through, store the first day's data
        if i_day == 1 : 
            rh_dlts.store_day(rh, current=False)
        
        # load next day for all variables
        ds.next()
    
    ds.close()