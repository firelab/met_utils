"""
Support for image toolchain and user-defined operators.

This module defines support for writing user-defined operators on numpy
ndarray objects. The operators may or may not maintain state. Iterators 
apply the operator over the input array one element at a time. Users may
chain operators together. An operator is defined as code which produces an
output array given zero or more input arrays. This implements the "pull" 
strategy for image computation.
"""

import numpy as np
from abc import ABCMeta, abstractmethod

class ArrayOperator (object) :
    pass
    
class CellArrayOperator (ArrayOperator) :
    """Performs the operation on the array on a cell-by-cell basis"""
    def apply(self, index) : 
        pass
    
class FunctionArrayOperator (CellArrayOperator) : 
    """Applies a function to individual cells of an array.
    
    In essense, this adapts a standalone function to the processing of 
    array data. Primarily, it maps the input ndarrays to function arguments.
    """
    pass
    
    
class IndividualStateOperator (CellArrayOperator): 
    """Applies a function which maintains per-cell state information
    
    Adapts a class which maintains state information between calls. In 
    this case, we want to ensure that each cell has its own copy of the 
    state information which can vary independently of the others. We also 
    want to make sure that classes written for point calculations are not 
    mistakenly accumulating state for adjacent cells.
    
    We accomplish this by permitting a subclass to define a dtype which 
    represents the necessary state information at each cell. We also allow
    the subclass to specify how many timesteps we need to recall state for.
    This class takes care of ensuring that a numpy array of the appropriate shape
    is allocated, and provides the machinery to store and retrieve state.
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, shape) : 
        dtype = self._get_history_dtype()
        self.hist_depth = self._get_history_depth()
        shape = shape + (self.hist_depth,)
        self._history = np.recarray(shape, dtype=dtype)
        
        self._i_history = 0
        
        self._init_history()
        
    @abstractmethod
    def _get_history_dtype(self) :
        """Returns the dtype used to store state information
        
        The subclass needs to override this method to provide a description
        of the data which needs to be maintained. It should create and return
        a dtype object suitable for its own use.
        """
        pass
        
    @abstractmethod
    def _get_history_depth(self) :
        """Returns an integer declaring how much history is necessary.
        
        When the object is created, the client specifies the size of the 
        grid and the subclass specifies the length of the "history" dimension. 
        These are combined to create a numpy array of the shape: 
            shape + (history_depth,)
            
        (i.e., the history_depth is appended as the last dimension to the shape
        provided when the object is created.)
        """
        pass
        
    @abstractmethod
    def _init_history(self): 
        """Subclass provides a method to initialize the history array.
        
        The history array is created using numpy.recarray, which doesn't initialize
        to anything. This method is called once before any image processing 
        begins. At the conclusion of this method, the subclass should be ready 
        to begin accumulating history.
        """
        pass
 
        
class NdarrayRoundRobin(np.ndarray) : 
    def __array_finalize__(self, obj) :
        self.__i_history = 0
        self.histdim   = len(obj.shape) - 1
        self.obj = obj
        
    def next(self) : 
        self.__i_history = (self.__i_history + 1) % self.obj.shape[self.histdim]
        
    def __recalc_index(self, index) : 
        """Intercept and recompute the history dimension"""
        i_new = (index[self.histdim] + self.__i_history) % self.obj.shape[self.histdim]
        if self.histdim == 0 : 
            index = (i_new,) + index[1:]
        elif self.histdim == len(self.obj.shape)-1 :
            index = index[:-1] + (i_new,)
        else :
            index = index[:self.histdim] + (i_new,) + index[self.histdim+1:]
 
        return index
                
    def __getitem__(self, index) : 
        return self.obj[self.__recalc_index(index)]
        
    def __setitem__(self, index, val) :
        self.obj[self.__recalc_index(index)] = val