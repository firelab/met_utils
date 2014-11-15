from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.ma as ma
import astropy.units as u
import astropy.coordinates as c

class SamplingFunction (object) : 
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_index(self, unit_val) : 
        pass
        
class LinearSamplingFunction ( SamplingFunction ) :
    def __init__(self, scale, offset=0) :
        self.scale = scale
        self.offset = offset
        
    def get_index(self, unit_val) : 
        return np.trunc(unit_val*self.scale + self.offset).astype(np.int)
        
class LongitudeSamplingFunction (SamplingFunction) : 
    E_ROT = 360*u.deg/(1*u.day)
    """earth angular velocity around own axis (approximate)"""
    
    def __init__(self, daily_samples, time_of_day) :
        self.daily_samples = daily_samples
        self.time_of_day = time_of_day
        self.time_angle = c.Angle( (time_of_day /(24*u.hour)) * 360 * u.deg)
        sample_interval = (1./daily_samples) * u.day
        self.time_to_sample = LinearSamplingFunction( 1/sample_interval )
        
    def get_index(self, unit_val) : 
        """converts longitude to sample"""
        lon_angle = c.Angle((-unit_val) + self.time_angle).wrap_at(360*u.deg) 
        return self.time_to_sample.get_index(lon_angle/self.E_ROT)

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
                  template=None, ref_time=13*u.hour, sequential=True):
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
            self.__init_buffer()
            

    def __init_buffer(self) : 
        self.load_day(1)
        self.cur_day = 2  
        
    def __init_lons(self) :
        """precomputes indices into 2-day buffer"""
        tmp = LongitudeSamplingFunction(self.diurnal_window, self.ref_time)
        self.i_ref = tmp.get_index(self.lons)   
        
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

        
    def next(self, data=None) : 
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
        
        if data == None : 
            start = self.cur_day * self.diurnal_window
            end   = start + self.diurnal_window
            self.buffer[i_today] = self._get_src_data(start,end)
        else : 
            # if the user provided the next day's data, use it.
            self.buffer[i_today] = data
        
        self.cur_day += 1
        
    def mean(self) : 
        return self.buffer_masked.mean(axis=self.time_axis)
        
    def max(self) : 
        return self.buffer_masked.max(axis=self.time_axis)
        
    def min(self): 
        return self.buffer_masked.min(axis=self.time_axis)
        
    def ref_val(self) : 
        """returns the variable's instantaneous value at the reference time
        
        The time sample selected depends on the longitude of the cell. 
        """
        result = np.empty( (len(self.i_ref),), dtype=self.source.dtype )
        
        # 2nd day is the "current day".
        i_ref_buf = self.i_ref + self.diurnal_window
        for i in range(len(i_ref_buf)) : 
            result[i] = self.buffer[i, i_ref_buf[i]]
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
            
        return result   
        
class DLTSGroup (object) : 
    """manages a group of related DiurnalLocalTimeStatistics objects
    
    Consolidates the management of DLTS objects.
    """
    def __init__(self) : 
        self.group = {} 
        
    def add(self, name, dlts) : 
        if len(self.group)  == 0 : 
            self.template = dlts
        self.group[name] = dlts
        
    def next(self) : 
        for k,v in self.group : 
            v.next() 
            
    def create(self, name, source) : 
        """creates a new dlts object, using the template
        
        This method avoids the need to call the computationally expensive 
        DiurnalLocalTimeStatistics.__init_lons() method when successive variables
        have exactly the same geospatial sampling. After adding at least one 
        DLTS object, you may call this method to use the first DLTS object 
        as a template. You must have called add() at least once prior to calling
        this function.
        """
        self.group[name] = DiurnalLocalTimeStatistics(source,template=self.template)
        return self.group[name]
    
    def get(self, name) : 
        return self.group[name]
        
        