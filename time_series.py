"""Abstracts out a time series which may be spread across multiple netCDF files."""
import netCDF4 as nc
import numpy as np
from collections import namedtuple, OrderedDict
import datetime as dt


def get_scalar_time(dataset, time_dim, datetime, unit_str=None) : 
    """returns scalar representation of the instant in time.
    
    Convenience wrapper for netCDF4.date2num, which ensures that the 
    conversion to a scalar representation of time is performed as specified
    in the netCDF file.
    
    The netcdf Dataset must be opened, the name of the time dimension and 
    the particular time of interest must be specified. It is expected that the 
    coordinate variable exists and has a units attribute which specifies 
    how the conversion is to take place. If the unit string is missing or
    known to be wrong, the caller may explicitly override it.
    
    The returned value is a floating point number"""
    if unit_str is not None : 
        units = unit_str.format
    else : 
        units = dataset.variables[time_dim].units
        
    return nc.date2num(datetime, units)
    
def get_calendar_time(dataset, time_dim, scalar_time, unit_str=None) : 
    """returns calendar representation of the specified scalar time
    
    Convenience wrapper for netCDF4.num2date, which ensures that the 
    conversion to a scalar representation of time is performed as specified
    in the netCDF file.
    
    The netCDF Dataset must be opened, the name of the time dimension and 
    scalar representation of the time of interest must be specified. It is 
    expected that the coordinate variable exists and has a units attribute which 
    specifies how the conversion is to take place. If the unit string is 
    missing or known to be wrong, the caller may explicitly override it.
    
    The returned value is a datetime object."""
    if unit_str is not None : 
        units = unit_str
    else :
        units = dataset.variables[time_dim].units
        
    return nc.num2date(scalar_time, units)


CacheObj = namedtuple('CacheObj', ['filename','time','dataset','unit_str'])

class Cache (object) : 
    """manages a cache of tuples (of type CacheObj) where lookup can be 
    performed on more than one element of the tuple"""
    def __init__(self) : 
        self._by_filename = {} 
        self._by_time     = {} 
        self._by_dataset  = {}
        
    def add(self, filename, time, dataset, unit_str=None) : 
        target = CacheObj(filename, time, dataset, unit_str) 
        self._by_filename[filename] = target
        self._by_time[time] = target
        self._by_dataset[dataset] = target
        
    def get(self, filename=None, time=None, dataset=None) : 
        """retrieves a cache object by filename, time, or dataset"""
        if filename is None : 
            if time is None : 
                target = self.get_by_dataset(dataset)
            else :
                target = self.get_by_time(time)
        else : 
            target = self.get_by_filename(filename)
        return target
        
        
    def remove(self, filename=None, time=None, dataset=None) : 
        """removes a cache object, specifying filename, time, or dataset"""
        target = self.get(filename, time, dataset)
        
        del self._by_filename[target.filename]
        del self._by_time[target.time]
        del self._by_dataset[target.dataset]
        
    def change_unit_str(self, unit_str, filename=None, time=None, dataset=None) : 
        """changes the unit string associated with the specified cache object
        
        The cache object may be specified by filename, time, or dataset"""
        target = self.get(filename, time, dataset)
            
        self.add(target.filename, target.time, target.dataset, unit_str)
        
    def get_by_filename(self, filename) : 
        return self._by_filename[filename]
        
    def get_by_time(self, time) : 
        return self._by_time[time]
        
    def get_by_dataset(self, dataset) : 
        return self._by_dataset[dataset]
        
    def filename_in(self, filename) : 
        return filename in self._by_filename
        
    def dataset_in(self, dataset) :
        return dataset in self._by_dataset
        
    def time_in(self, time) : 
        return time in self._by_time
        
    def get_all_datasets(self) : 
        return self._by_dataset.keys()
    
    def get_any_dataset(self) : 
        return self._by_dataset.keys()[0]
        

class TimeSeries (object) : 
    """Multi-file netcdf capability along a time axis.
    
    This functionality is a limited subset of the functionality offered by
    MFDataset. The primary difference is that MFDataset joins files along
    the unlimited dimension and the time dimension in my files is finite.
    
    This class does not expect that the scalar values for time monotonically 
    increases across files. For instance, annual files could each have a 
    units string "days since YYYY-01-01", where YYYY is the year for which
    the file contains data. Each file in the series would have scalar time 
    values between 0 and 366.
    """
    
    def __init__(self, file_pattern, time_dim, unit_pattern=None) : 
        """Initializes a TimeSeries object
        
        This object requires knowledge of the file naming pattern (to open
        the correct file) and the structure of the dimensions within the file
        (to return the correct range within the file). If the entire time 
        series is contained within a single file, specify the file name as the
        file pattern, otherwise specify a format string which will produce 
        the correct file name when given a datetime object.
        
        If the time dimension is known not to contain a units attribute,
        the unit_pattern must be specified. This pattern must be a format
        string which produces the correct units string when provided with
        a datetime object.
        """
        self.file_pattern = file_pattern
        self.time_dim     = time_dim
        self.unit_pattern = unit_pattern
        self.file_cache   = Cache()
        
    
    def get_file_name(self, datetime) : 
        """returns the name of the file containing the specified datetime
        
        The date/time/datetime object provided must be compatible with 
        the file_pattern string specified at instantiation"""
        return self.file_pattern.format(datetime)
        
    def get_dataset(self, datetime, **kwargs) : 
        """returns a netCDF Dataset object for the specified point in time.
        
        The date/time/datetime object provided must be compatible with the 
        file_pattern string specified at instantiation. If the file has not 
        previously been opened, the new Dataset object is added to the cache. 
        Otherwise, the previously opened Dataset object is returned.
        
        Any keyword arguments passed to this function are relayed to the 
        Dataset constructor, so you may control how the file is opened.
        """
        fname = self.get_file_name(datetime)
        
        if self.file_cache.filename_in(fname) : 
            retval = self.file_cache.get_by_filename(fname)
        else : 
            retval = nc.Dataset(fname, **kwargs)
            self.file_cache.add(fname, datetime, retval)

        return retval
        
    def close(self) : 
        """closes all cached datasets and clears the cache"""
        for ds in self.file_cache.get_all_datasets() :
            ds.close()
        self.file_cache = Cache()
        
    def get_index(self, datetime, dataset=None) : 
        """computes the index along the time axis corresponding to the specified time
        
        The caller may optionally specify the dataset if known, otherwise it 
        is looked up."""
        if dataset is None : 
            dataset = self.get_dataset(datetime)
        units = None
        if self.unit_pattern is not None : 
            units = self.unit_pattern(datetime)
        scalar = get_scalar_time(dataset, self.time_dim, datetime, units)
        return np.floor(scalar).astype(np.int)
        
    def get_cal_time(self, scalar_time, dataset)  :
        """computes the calandar time corresponding to the provided scalar
        time in the provided dataset.
        
        This is essentially the inverse operation of get_location()."""
        
        # unit string may be stored in the cache
        # or, we can use "a" datetime object to calculate it
        # all datetime objects associated with the same file
        # should result in the same unit string.        
        cache_obj = self.file_cache.get_by_dataset(dataset)
        datetime = cache_obj.time
        units = cache_obj.unit_str
        if self.unit_pattern is not None : 
            units = self.unit_pattern(datetime)
            
        return get_calendar_time(dataset, self.time_dim, scalar_time,units) 
        
        
    def get_location(self, datetime) : 
        """looks up a netCDF dataset and index corresponding to the specified time"""
        dataset = self.get_dataset(datetime)
        index   = self.get_index(datetime, dataset=dataset)
        return (dataset, index)
        
    def get_interval(self, start, delta) : 
        """returns locations in the netcdf file(s) for the endpoints of the 
        specified interval"""
        stop = start + delta
        start_loc = self.get_location(start)
        stop_loc  = self.get_location(stop)
        return (start_loc, stop_loc)
        
    def any(self) :
        return self.cache.get_any_dataset()
         
class IntegerInterval (object) : 
    """Generator class which remembers a start, stop, and skip
    
    Class can be used to repeatedly iterate over the same window"""
    def __init__(self, *args) : 
        if len(args == 1) : 
            self.stop = args[0]
            self.start = 0
            self.step  = 1
        else : 
            self.start = args[0]
            self.stop  = args[1]
            if len(args) == 3 : 
                self.step = args[2]
            else : 
                self.step = 1
                
    def interval(self) : 
        for i in range(self.start, self.stop, self.step) :
            yield i

class DateTimeInterval (object) : 
    def __init__(self, start, stop, step=dt.timedelta(days=1)) :
        self.start = start
        self.stop  = stop
        self.step  = step

    def length(self) : 
        return int((self.stop - self.start).total_seconds()/self.step.total_seconds())
        
    def interval(self) : 
        i = 0 
        date_i = self.start
        while date_i < self.stop : 
            yield date_i
            i += 1
            date_i = self.start + i*self.step

class AnnualInterval(object) : 
    """An object which advances exactly one year for each interval
    
    This generator class produces datetime.date objects for the Jan 1
    of each year in the series."""
    def __init__(self, start, stop, step=1) : 
        """configure the range of years over which to iterate"""
        self.start = start
        self.stop  = stop
        self.step  = step
        
    def length(self) : 
        return (self.stop - self.start)/self.step
        
    def interval(self)  :
        for year in range(self.start, self.stop, self.step) : 
            yield dt.date(year, 1, 1)
            
class IntraAnnualInterval(object) : 
    """An object which iterates over a list of seasons, and can tell the 
    length of the current season"""
    def __init__(self, interval_dict, year=None) : 
        """season_dict is an ordered dictionary, where the key is the 
        name of the interval and the value is the date object indicating the 
        first day of the interval.
        If the year is given, the year field of each of the date objects
        is set to it."""
        self.interval_dict = interval_dict
        if year is not None : 
            self.set_year(year)
        else : 
            self.year = interval_dict.values()[0].year
            self._compute_lengths()
    
    def set_year(self, year) : 
        fixed = OrderedDict()
        delta_yr = None
        for k,v in self.interval_dict.iteritems() : 
            if delta_yr is None : 
                delta_yr = year - k.year
            k_prime = dt.date(k.year+delta_yr,k.month, k.day)
            fixed[k_prime] = v
        self.interval_dict = fixed
        self.year = year
        self._compute_lengths()
        
    def _compute_lengths(self) :
        lengths = {}
        interval_list = self.interval_dict.items()
        for i in range(len(interval_list)-1) :
            name = interval_list[i][0] 
            cur = interval_list[i][1]
            nxt = interval_list[i+1][1]
            lengths[name] = nxt-cur
            
        name = interval_list[-1][0]
        cur = interval_list[-1][1]
        nxt = interval_list[0][1]
        nxt_prime = dt.date(self.year+1, nxt.month, nxt.day)
        lengths[name] = nxt_prime - cur
        self.lengths = lengths
            
        
    def interval(self) : 
        """This generator returns the interval name, the starting date, and
        the length of the interval"""
        for k,v in self.interval_dict.iteritems() : 
            yield (k, v, self.lengths[k])

def monthly_intervals(year) : 
    months = OrderedDict() 
    
    months["January"] = dt.date(year,1,1)
    months["February"] = dt.date(year,2,1)
    months["March"]    = dt.date(year,3,1)
    months["April"]    = dt.date(year,4,1)
    months["May"]      = dt.date(year,5,1)
    months["June"]     = dt.date(year,6,1)
    months["July"]     = dt.date(year,7,1)
    months["August"]   = dt.date(year,8,1)
    months["September"]= dt.date(year,9,1)
    months["October"]  = dt.date(year,10,1)
    months["November"] = dt.date(year,11,1)
    months["December"] = dt.date(year,12,1)
    
    return IntraAnnualInterval(months)
    
def seasonal_intervals(year) : 
    seasons = OrderedDict()
    
    seasons['DJF'] = dt.date(year,12,1)
    seasons['MAM'] = dt.date(year+1,3,1)
    seasons['JJA'] = dt.date(year+1,6,1)
    seasons['SON'] = dt.date(year+1,9,1)
    
    return IntraAnnualInterval(seasons)

class LoopHandler(object)  :
    def pre(self, *args, **kwargs) : 
        pass
        
    def post(self, *args, **kwargs) : 
        pass
        
    def per_iteration(self, *args, **kwargs) :
        pass


class Looper (object) : 
    """Control logic to call handler code while looping over values 
    in one of the Interval classes."""
    def __init__(self, loop, handler) : 
        self.loop = loop
        self.handler = handler 
        
    def iterate(self) : 
        self.handler.pre()
        for i in self.loop.interval() : 
            self.handler.per_iteration(i)
        self.handler.post()
                
class NestedLoop (object) : 
    """Control logic to handle an inner and outer loop, each with their
    own handlers. The handlers must be written to understand the 
    arguments they are getting, based on whether they are for the
    inner or outer loop."""
    def __init__(self, outer, inner, outer_handler, inner_handler) : 
        self.outer = outer
        self.inner = inner
        self.outer_handler = outer_handler
        self.inner_handler = inner_handler
        
    def inner(self) : 
        for i in self.inner.interval() :
            yield i
    
    def outer(self) : 
        for i in self.outer.interval() : 
            yield i
            
    def iterate(self)  :
        self.outer_handler.pre()
        for i in self.outer() : 
            #self.outer_handler.per_iteration(i)
            self.inner_handler.pre(i)
            for j in self.inner() : 
                self.inner_handler.per_iteration(i,j)
            self.inner_handler.post(i)
        self.outer_handler.post()
            
class AnnualWindow(NestedLoop) : 
    """Outer loop iterates over years while inner loop iterates over days"""
    def __init__(self, y_start, y_stop, y_handler, d_start, d_stop, d_handler) : 
        """initializes inner and outer loops
        
        y_start, y_stop, y_handler initialize the outer loop for years
        
        d_start, d_stop, d_handler initialize the inner loop for the starting
        and stopping days"""
        inner = IntegerInterval(d_start,d_stop)
        outer = AnnualInterval(y_start,y_stop)
        super(AnnualWindow, self).__init__(outer,inner, y_handler, d_handler)
        