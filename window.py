# -*- coding: utf-8 -*-
"""
Implements moving window functionality along one dimension of an array-like.

Meant for very large arrays stored mostly on disk, where sequential reads
must be retained for a defined number of simulation steps. A limited set of 
operations is offered. 
"""
import numpy as np
import astropy.units as u

class MovingWindow (object) : 
    """Implements a moving window into numeric data
    
    User specifies the shape of the buffer they are windowing into, 
    the axis along which they are windowing, and the size of the window
    in samples/indices. The shape provided is expected to include the 
    window axis. This object does not handle masked arrays.
    """
    def __init__(self, shape, window_axis, window_size, dtype=np.float32, 
                  initial_value=0, unit=None) :
        self.window_size = window_size 
        self.window_axis = window_axis
        shape = list(shape)
        shape[window_axis] = window_size
        self.shape = tuple(shape)
        self.buffer = np.empty(self.shape, dtype=dtype)
        self.next_slice = 0 
        self.index_template = ( slice(None, None, None), ) * len(shape)
        self.num_puts = 0 
        self.unit = unit

        if unit is None : 
            self.buffer[:] = initial_value
        else : 
            self.buffer[:] = u.Quantity(initial_value, unit=unit)


    def put(self, data) : 
        """puts the next slice of data into the buffer"""
        #convert if necessary
        if self.unit is not None : 
            data = u.Quantity(data, unit=self.unit)
            
        index = list(self.index_template)
        index[self.window_axis] = self.next_slice
        self.buffer[index] = data
        self.next_slice = (self.next_slice +1) % self.window_size
        self.num_puts += 1
        
    def get(self, i) :
        """retrieves a slice of data
        
        The most current slice (last one "put()") has the highest index (window_size-1).
        The oldest slice has index 0. The window will "wrap". It is up to the caller
        to not ask for a slice outside the range [0..window_size-1].
        """
        oldest = self.next_slice
        i = (oldest + i) % self.window_size
        index = list(self.index_template)
        index[self.window_axis] = i
        
        result = self.buffer[index]
        if self.unit is not None : 
            result = result * self.unit
        return result
        
    def mean(self) : 
        """calculates the mean of the current window along the window axis
        
        The returned array is missing the window axis.
        """
        result = np.mean(self.buffer, axis=self.window_axis)
        if self.unit is not None : 
            result = result * self.unit
        return result
        
    def sum(self) : 
        """calculates the total of the current window along the window axis"""
        result = np.sum(self.buffer, axis=self.window_axis)
        if self.unit is not None : 
            result = result * self.unit
        return result
        
    def ready(self) : 
        """returns whether we are ready to compute statistics
        
        Unless you like "edge effects", don't compute statistics until the 
        buffer has been filled. You have been warned.
        """
        return self.num_puts >= self.window_size
        
class SequenceWindow (MovingWindow) :
    """A moving window which accumulates a history of cells which meet a criteria
    
    A SequenceWindow deals in slices of boolean type. A true value in a cell 
    means that the criteria has been met, a false value means that the criteria
    has not been met. The class provides the ability to analyze the history
    present in the buffer, for each cell. 
    """
    def __init__(self, shape, window_axis, window_size,  
                  initial_value=False) :
        super(SequenceWindow, self).__init__(shape, window_axis, window_size,
                dtype=np.bool, initial_value=initial_value)        
        
    def last_run_length(self) : 
        """calculates the "run length" at each cell
        
        Starting with the last slice added and proceeding to previous slices, 
        count the number of sequential true values. Quit counting when a false
        value is encountered. A False value in the current slice means the cell
        gets a zero value.
        """
        run = None
        done = None
        for i in range(self.window_size-1, -1, -1) : 
            cur = self.get(i)
            
            if run is None : 
                run = np.zeros( cur.shape, dtype=np.int )
                done = np.zeros( cur.shape, dtype=np.bool )
            
            # gate off the cells which have a False value
            done = np.logical_or(done, np.logical_not(cur))
  
            # increment cells which are not done
            run[np.logical_not(done)] += 1
            
        return run
                
            
        
    def all_runs(self) : 
        """counts all runs in the buffer
        
        The value returned for each cell is a weighted sum of the contiguous
        slices containing a True value for that cell. The weighting is designed 
        to return a larger number for sequential runs of True values than it
        would return for the same number of True values separated by False
        values.
        
        This algorithm came from [1].
        
        [1] Flannigan, M. D. & Harrington, J. B. A Study of the Relation of 
        Meteorological Variables to Monthly Provincial Area Burned by Wildfire 
        in Canada (1953–80). J. Appl. Meteor. 27, 441–452 (1988).
        """
        run = None
        cum_run = None
        for i in range(self.window_size) : 
            cur = self.get(i)
            
            if run is None : 
                run = np.zeros( cur.shape, dtype=np.int)
                cum_run = np.zeros( cur.shape, dtype = np.int)
                
            # increment run counter for all True cells
            run[ cur ] += 1
            
            # reset run counter for all false cells
            run[np.logical_not(cur)] = 0
            
            # accumulate 
            cum_run += run
            
        return cum_run
            