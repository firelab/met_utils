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
            forcing = self.get_forcing()
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
        
    def get_timestep(self) :
        """caches/calculates the timestep from the info in the file""" 
        if not hasattr(self, "_timestep") : 
            forcing = self.get_forcing()
            times = forcing.variables['time'][:2] * u.s
            self._timestep = (times[1] - times[0])
        return self._timestep
        
    def register_variable(self, varname) :
        """adds "varname" to the list of variables to track""" 
        if not hasattr(self, "_buffers") : 
            self._buffers = {} 
        
        forcing = self.get_forcing()
        ncvar = forcing.variables[varname]
        time_axis = ncvar.dimensions.index('tstep')
        self._buffers[varname] = q.DiurnalLocalTimeStatistics(
            forcing.variables[varname], time_axis,
            self.get_timestep(), self.get_longitudes())
        return self._buffers[varname]

    def next(self) :
        """advance all of the registered variables to the next day"""
        for key in self._buffers : 
            self._buffers[key].next()
        
    def get_variable(self, varname) :
        """returns the DiurnalLocalTimeStatistics object associated with the registered varname""" 
        return self._buffers[varname]
                
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
    
    # register required variables so they are tracked
    qair = ds.register_variable("Qair")
    tair = ds.register_variable("Tair")
    pres = ds.register_variable("Psurf")
#    rainf = ds.register_variable("Rainf")
#    swdown = ds.register_variable("SWdown")
    
    # setup netcdf dimensions in output file
    ds.copyDimension("land")
    ds.copyDimension("y")
    ds.copyDimension("tstep")
    ds.createDimension("days", 365)
    
    # create netcdf variables to hold indices in output file
    daylength = ds.create_variable("daylength", ("days","y"), np.float32)
    daylength.title = "Day Length"
    daylength.units = "hours"
    
    rh = ds.create_variable("rh", ("tstep","land"), np.float32)
    rh.title = "Relative Humidity"
    rh.units = "percent"
    
    rh_max = ds.create_variable("rh_max", ("days","land"), np.float32)
    rh_max.title = "Maximum RH" 
    rh_max.units = "percent"
    
    rh_min = ds.create_variable("rh_min", ("days","land"), np.float32)
    rh_min.title = "Minimum RH"
    rh_min.units = "percent"
    
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
        daylength[i_day,:] = (ds.get_daylength_by_lat(day)).to(u.hour)
        
        # calculate RH
        
        # load next day for all variables
        ds.next()
    
    ds.close()