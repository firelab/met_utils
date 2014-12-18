from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.ma as ma
import astropy.units as u
import astropy.coordinates as c
import scipy.spatial as sp

# cannot compare "unit" objects to python's None. 
# creating an "unspecified" fundamental unit to signify that 
# no unit object has been specified
unspecified_u = u.def_unit("unspecified")

class SamplingFunction (object) : 
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_index(self, unit_val) : 
        pass
        
class LinearSamplingFunction ( SamplingFunction ) :
    """y=mx+b, where "x" is a unitted value, and y is index into array
    
    When creating the object, "m" == scale and "b" == offset.
    """
    def __init__(self, scale=None, offset=0, x_zero=None, minmaxbin=None,
                 includemax=True) :
        """Initializes a LinearSamplingFunction object.
        
        Three possible ways to initialize this object include specifying a 
        scale factor with either a precalculated offset or the "x" associated with
        y=0, or specifying a min/max/number of bins tuple.
        """
        self.includemax = False

        if scale is not None : 
            self.scale = scale

        if minmaxbin is not None : 
            minval, maxval, num_bins = minmaxbin
            num_bins = np.trunc(num_bins).astype(np.int)
            self.scale = num_bins / (maxval - minval)
            self.includemax = includemax
            self.maxval = maxval
            self.maxbin = num_bins-1
            x_zero = minval

        if x_zero is not None:  
            offset = -(self.scale * x_zero)
        self.offset = offset
        
    def get_index(self, unit_val) :
        index = u.Quantity(unit_val*self.scale + self.offset, unit=u.dimensionless_unscaled)
        int_index = np.trunc(index).astype(np.int)
        if self.includemax : 
            int_index = np.where(unit_val == self.maxval, self.maxbin, int_index)
        return int_index
        
    def get_unit_val(self, i) : 
        """calculates the physical unit given the integer index"""
        unit_val = (i-self.offset)/self.scale
        return unit_val
        

class TimeSinceEpochFunction ( SamplingFunction ) : 
    """calculates an index given a timestamp
    
    Create this object with an epoch, which is the timestamp of the first 
    element in the array. Supply the interval between each sample as well.
    """
    def __init__(self, interval, epoch) :
        self.interval = interval
        self.epoch  = epoch 
        
    def get_index(self, time) : 
        index = ((time - self.epoch).to(self.interval.unit) / self.interval)
        index = index.to(u.dimensionless_unscaled)
        return np.trunc(index).astype(np.int)
        
class LongitudeSamplingFunction (SamplingFunction) : 
    E_ROT = 360*u.deg/(1*u.day)
    """earth angular velocity around own axis (approximate)"""
    
    def __init__(self, daily_samples, time_of_day) :
        """initializes a longitude sampling function
        
        daily_samples should be a dimensionless number specifying the number 
        of samples per day. time_of_day indicates the reference time of day in 
        local time.
        """
        self.daily_samples = daily_samples
        self.time_of_day = time_of_day
        self.time_angle = c.Angle( (time_of_day /(24*u.hour)) * 360 * u.deg)
        sample_interval = (1./daily_samples)*u.day
        self.time_to_sample = LinearSamplingFunction( 1/sample_interval )
        
    def get_index(self, unit_val) : 
        """converts longitude to sample"""
        lon_angle = c.Angle((-unit_val) + self.time_angle).wrap_at(360*u.deg) 
        return self.time_to_sample.get_index(lon_angle/self.E_ROT)

class IdentitySamplingFunction (SamplingFunction) :
    def get_index(self, unit_val) : 
        """just passes unit_val through"""
        return unit_val 

class CoordinateVariableSamplingFunction (SamplingFunction)  :
    """looks up index based on value of a coordinate variable
    
    Named after the NetCDF concept of a "coordinate variable", which is a 
    1D variable the length of an axis, this sampling function looks up the 
    index based on the position of the given value in a "coordinate variable" 
    array
    
    Note that currently this only works if the exact value you are searching
    for is in the coordinate variable.
    """ 
    def __init__(self, cv) : 
        self.cv = cv
        
    def get_index(self, unit_val) : 
        return self.cv.index(unit_val)   
            
class OrthoIndexer (SamplingFunction)  : 
    """Combines multiple SampleFunctions into one
    
    This implementation assumes that all array dimensions are part of the 
    domain, and that all axes are orthogonal. The number of sampling functions
    is the same as the number of unitted values provided, which is the same
    as the number of indices returned.
    
    The conversion between indices and unit vals happens for single sets of 
    coordinates, not for sequences of coordinates. This is different
    than the above.
    """
    def __init__(self, sample_functions) : 
        """Initializes an NDindexer given a list of sampling functions"""
        self.sample_functions = sample_functions
        
    def get_index(self, unit_val) : 
        """returns a tuple of indices 
        """
        ret = np.empty( (len(self.sample_functions),), dtype=int)
        for i in range(len(self.sample_functions)) : 
            ret[i] = self.sample_functions[i].get_index(unit_val[i])
        return tuple(ret) 

    def get_unit_val(self, index) : 
        """returns a tuple of unit_vals 
        """
        ret = np.empty( (len(self.sample_functions),))
        for i in range(len(self.sample_functions)) : 
            ret[i] = self.sample_functions[i].get_unit_val(index[i])
        return tuple(ret) 


class UnitIndexNdArray (object) : 
    """provides single-element access to ndarray using unitted quantities"""
    
    def __init__(self, array, indexer) : 
        """combines an array and an indexer"""        
        self.array = array
        self.indexer = indexer
        
    def get(self, index) : 
        """returns single value from array"""
        return self.array[self.indexer.get_index(index)]
        
    def put(self, index, value) : 
        """puts single value in array"""
        self.array[self.indexer.get_index(index)] = value
        
    def inc(self, index, value=1) : 
        """increments value at index position"""
        self.array[self.indexer.get_index(index)] += value

class LookupTable(object) :
    """Performs 1D nearest neighbor resampling""" 
    def __init__(self, bin_min, bin_width, values) : 
        """bins, values define the lookup table
        
        bin_min represents the center of the first bin
        bin_width represents distance between bins
        value is a parallel array representing the value for a bin
        """
        self.bin_min = bin_min
        self.bin_width = bin_width
        self.values = values
        self.indexer = LinearSamplingFunction(1./bin_width, 
                          -(bin_min/bin_width).to(u.dimensionless_unscaled))
        
        
    def get(self, x) : 
        """retrieves a value based on which bin "x" falls into"""
        i = self.indexer.get_index(x)
        return self.values[i]
    

class DiurnalLocalTimeStatistics (object) : 
    """Computes 24-hour summary statistics based on local time.
    
    Given a (possibly global) dataset referred to a non-local time like UTC,
    compute 24 hour statistics using local time and assign the result to a
    UTC date. The statistics are referenced to a common local time for all cells. 
    The time associated with the statistic is the UTC date on which the common
    local time falls.
    
    For example, if the reference time is 1300 local, this means the statistics 
    are computed for 13 hours after the local midnight. The offset between local
    midnight and UTC midnight is computed using the longitude of the cell, not 
    the time zone in which the cell resides.
    
    Statistics computed for a particular UTC date are drawn from data taken
    in the 24-hour window prior to the reference local time on that 
    UTC date.
    
    The data source is expected to be a numpy array-like with one axis being 
    time. Instances of this class may be configured to compute statistics 
    directly on the data source (i.e., when the data reside in memory), or 
    an internal buffer may be used (i.e., when the data are on disk). Index 0
    of the time dimension is assumed to represent UTC midnight of the start
    of the dataset. Samples along the time axis are assumed to be equally
    spaced. Diurnal statistics may be computed for any integer number of UTC
    days from the start of the dataset.
    
    The class provides a "next()" method to compute the statistics for the
    next day. This is provided both for convenience and for efficiency when 
    the buffer mode is used. Random access mode will always have to load two
    days worth of data from the file, whereas sequential access allows 
    a single day's data to be read. Because the previous day's data are always
    required, the computation of statistics for "day 0" is not allowed. 
    """
    def __init__(self, source, time_axis=None, timestep=None, lons=None, 
                  template=None, ref_time=13*u.hour, sequential=True, 
                  unit=unspecified_u):
        """Wraps a data source to use as the basis of daily statistics.
        
        The caller must specify a numpy array-like data source, the index of the
        time axis of that data source, the timestep and longitude for each cell.
        This class assumes a 1D "compressed axes" storage pattern, where the 
        longitude of every cell must be individually specified in a parallel
        array.
        
        Everything but the source numpy array-like may be specified using a 
        template, which is just a previously created DiurnalLocalTimeStatistics
        object. This can save some expensive computations when initializing the 
        mask. However, everything about the source must be the same as the 
        source in the template (shape, longitudes of each cell, etc.) 
        
        Sequential access is assumed unless otherwise specified. The local 
        reference time for statistics is 1300 hours.
        """
        self.source = source
        self.i_buf_template = ( slice(None,None,None), ) * len(source.shape)
        self.buffer = None
        self.cur_day = None
        self.unit = unit        

        # we either init most things off of a template, or we compute them 
        # from scratch
        if template == None : 
            self.diurnal_window = ((24*u.hour)/timestep).to(u.dimensionless_unscaled).astype(np.int)
            self.lons = lons
            self.time_axis = time_axis
            self.timestep = timestep
            self.ref_time = ref_time
            self.__init_lons()
        else : 
            self.diurnal_window = template.diurnal_window
            self.lons = template.lons
            self.time_axis = template.time_axis
            self.timestep = template.timestep
            self.ref_time = template.ref_time
            # don't call __init_lons(). Too expensive. Just copy.
            self.i_ref = template.i_ref
            self.mask  = template.mask
        
        # if user wants sequential access, initialize buffer
        if sequential : 
            self._init_buffer()
            

    def _init_buffer(self) : 
        self.load_day(1)
        self.cur_day = 2  
        
    def __init_lons(self) :
        """precomputes indices into 2-day buffer"""
        tmp = LongitudeSamplingFunction(self.diurnal_window, self.ref_time)
        
        # makes a lookup table for 0.5 deg cells
        lut = LookupTable(-179.75 *u.deg, 0.5*u.deg,
                          tmp.get_index(np.arange(-179.75,180,0.5)*u.deg))
        self.i_ref = lut.get(self.lons)   
        
        # create a mask array
        mask_shape = np.copy(self.source.shape)
        mask_shape[self.time_axis] = 2*self.diurnal_window
        self.mask = np.ones( mask_shape, dtype=np.bool) # mask everything
        # now enable the samples used in the calculation
        it = np.nditer(self.mask, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished  :
            # this only works because we assume 1d compressed spatial + time
            i_time = it.multi_index[self.time_axis]
            i_lon  = it.multi_index[not self.time_axis]
            
            it[0] = not ((i_time > self.i_ref[i_lon]) and (i_time <= (self.i_ref[i_lon] +  self.diurnal_window)))
            it.iternext()

    
    def _get_src_data(self,time_start,time_end):
        """given a time slice, return the source data"""
        timeslice = slice(int(time_start), int(time_end))
        i_buf = list(self.i_buf_template)
        i_buf[self.time_axis] = timeslice
        return np.copy(self.source[i_buf])
                
    def load_day(self, day) : 
        """loads the specified day's data from the source into the buffer
        
        This method always loads 2 days worth of data: the previous day and 
        the current day. The size of 1 day's data is "self.diurnal_window"
        Loading the first day (day==0) is always illegal. 
        """
        start = (day-1) * self.diurnal_window
        end = start + (2 * self.diurnal_window)
        self.buffer = self._get_src_data(start,end)
        self.buffer_masked = ma.array(self.buffer, mask=self.mask)

    def get_num_landpts(self) : 
        return len(self.lons)
        
        
    def next(self) : 
        """loads the next day's data into the buffer
        
        Shifts the "current day's" data into the "yesterday" position in the
        buffer, then increments the current day counter and loads the new data
        in.
        
        This method shifts "buffer" instead of "buffer_masked" because it needs
        to move the data without moving the mask.
        """
        # construct the "yesterday" index
        yesterday_slice = slice(0,self.diurnal_window)
        i_yesterday = list(self.i_buf_template)
        i_yesterday[self.time_axis] = yesterday_slice

        # construct the "today" index
        today_slice = slice(self.diurnal_window, 2*self.diurnal_window)
        i_today     = list(self.i_buf_template)
        i_today[self.time_axis] = today_slice
        
        self.buffer[i_yesterday] = self.buffer[i_today]
        
        start = self.cur_day * self.diurnal_window
        end   = start + self.diurnal_window
        self.buffer[i_today] = self._get_src_data(start,end)
        
        self.cur_day += 1
        
    def mean(self, unitted=True) : 
        result = self.buffer_masked.mean(axis=self.time_axis).data
        if unitted and self.unit != unspecified_u : 
            result = result * self.unit
        return result
        
    def max(self, unitted=True) : 
        result = self.buffer_masked.max(axis=self.time_axis).data
        if unitted and self.unit != unspecified_u :
            result = result * self.unit
        return result
        
    def min(self, unitted=True): 
        result = self.buffer_masked.min(axis=self.time_axis).data
        if unitted and self.unit != unspecified_u : 
            result = result * self.unit
        return result
        
    def sum(self, unitted=True) : 
        result = self.buffer_masked.sum(axis=self.time_axis).data
        if unitted and self.unit != unspecified_u : 
            result = result * self.unit
        return result
        
    def ref_val(self, unitted=True) : 
        """returns the variable's instantaneous value at the reference time
        
        The time sample selected depends on the longitude of the cell. 
        """
        # a vector the size of the compressed geospatial axis
        result = np.empty( (len(self.i_ref),), dtype=self.source.dtype )
        
        # 2nd day is the "current day".
        i_ref_buf = self.i_ref + self.diurnal_window
        i_buf = [0,0]
        i_time = self.time_axis
        i_land = not i_time
        for i in range(len(i_ref_buf)) : 
            # buffer is 2D: timesteps x geospatial
            i_buf[i_time] = i_ref_buf[i]
            i_buf[i_land] = i
            result[i] = self.buffer[i_buf]
            
        if unitted and self.unit != unspecified_u : 
            result = result * self.unit
            
        return result
    
    def get_preceeding_day(self) : 
        """returns the record of instantaneous values for the day prior to the reference time
        
        Essentially this just copies all the unmasked values from the buffer to 
        the result array. The result is a square array with the time dimension
        containing 24 hours of samples. Samples which are after the local reference
        time, or which are more than 24 hours prior to the reference time have been 
        filtered out. Without knowledge of the UTC timestamp of the sample corresponding 
        to the local reference time, it is not possible to recover the UTC 
        time of any particular cell.
        """
        j = [0]*2
        i_result = [ slice(None, None, None) ] * 2
        i_lon = not self.time_axis
        i_time = self.time_axis
        resultshape = list(self.buffer.shape)
        resultshape[self.time_axis] = self.diurnal_window
        result = np.empty( resultshape, dtype = self.source.dtype)        
        
        for i in range(len(self.i_ref)) : 
            timeslice = slice(self.i_ref[i] + 1, self.i_ref[i] + self.diurnal_window + 1)
            j[i_lon] = i
            j[i_time] = timeslice
            i_result[i_lon] = i
            result[i_result] = self.buffer[j]
        
        # set the units of the data if we know them    
        if self.unit != unspecified_u : 
            result = result * self.unit
            
        return result 
        
    def get_utc_day(self, current_day=True) : 
        """Get the raw data for the current UTC day
        
        If you want the previous utc day's data, set current_day to False.
        """
        # construct the "today" index
        start = 0 
        if current_day : 
            start = self.diurnal_window
        day_slice = slice(start, start + self.diurnal_window)
        i_day     = list(self.i_buf_template)
        i_day[self.time_axis] = day_slice 

        result = self.buffer[i_day]
        
        # set the units of the data if we know them
        if self.unit != unspecified_u : 
            result = result * self.unit
            
        return result
        
    def get_buffer(self) : 
        """returns the buffer as a quantity object, if possible"""
        result = self.buffer
        if self.unit != unspecified_u : 
            result = result * self.unit
        return result
        
    def store_day(self, destination, current=True) : 
        """stores one day of data from buffer to destination buffer
        
        destination is assumed to have same shape as source."""
        day = self.cur_day - 1 # index of "today"
        if not current : 
            day -= 1 #index of yesterday
        
        start = day * self.diurnal_window
        end = start + self.diurnal_window
        timeslice = slice(int(start), int(end))
        i_buf = list(self.i_buf_template)
        i_buf[self.time_axis] = timeslice
        
        destination[i_buf] = self.get_utc_day(current)

        

class ComputedSourceDLTS (DiurnalLocalTimeStatistics) : 
    """A diurnal local time statistics class where the source is computed
    
    This class draws new data from an ever changing source variable. Whereas the
    parent class assumes the source value has a time axis which covers the
    entire domain of the simulation (and therefore we need to load in one day 
    at a time by advancing the index along the time axis), this class assumes 
    the source contains exactly one day's data. The user is responsible for 
    writing new values to the source before calling the next() method.
    
    You are explicitly allowed to assign new numpy arrays to the source attribute.
    The source you initially provide should have two days of data, in order to 
    initialize the internal 2-day buffer. The source buffer should be one day 
    big by the time you call next().
    """
    def _get_src_data(self, start, end) : 
        return self.source
        
           
                        
class DLTSGroup (object) : 
    """manages a group of related DiurnalLocalTimeStatistics objects
    
    Consolidates the management of DLTS objects.
    """
    def __init__(self) : 
        self.group = {} 
        self.template = None
        
    def add(self, name, dlts) : 
        """adds a named, pre-created DLTS object to the group."""
        if len(self.group)  == 0 : 
            self.template = dlts
        self.group[name] = dlts
        
    def next(self) : 
        """call next() method for each object in group"""
        for k in self.group : 
            self.group[k].next() 
            
    def create(self, name, source, unit) : 
        """creates a new dlts object, using the template
        
        This method avoids the need to call the computationally expensive 
        DiurnalLocalTimeStatistics.__init_lons() method when successive variables
        have exactly the same geospatial sampling. After adding at least one 
        DLTS object, you may call this method to use the first DLTS object 
        as a template. You must have called add() at least once prior to calling
        this function.
        """
        self.group[name] = DiurnalLocalTimeStatistics(source,
                template=self.template, unit=unit)
        return self.group[name]
        
    def create_computed(self, name, source, unit) : 
        """creates a computed dlts object using the template"""
        self.group[name] = ComputedSourceDLTS(source, 
            template=self.template, unit=unit)
        return self.group[name]
    
    def get(self, name) : 
        """fetches a named dlts object"""
        return self.group[name]
        
    def template_ready(self) :
        """are we ready to create objects using a template?""" 
        return self.template != None
        
        
