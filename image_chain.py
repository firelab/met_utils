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
    """Code which performs a calculation on one or more input arrays producing one or more output arrays
    
    
    """
    pass
    
class CellArrayOperator (ArrayOperator) :
    """Performs the operation on the array on a cell-by-cell basis
    
    This class requires the outputs and all the inputs to be coregistered. This 
    means that all of the inputs and outputs have the same shape. Values in 
    the inputs and outputs having the same index correspond. The 
    main function of this class is to iterate over the array, applying the 
    operation to each cell. The outputs are allocated by this class, and the 
    inputs are provided.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, inputs, output_keys, dtype=None) : 
        """Initializes a new CellArrayOperator with inputs and outputs.
        
        Parameters
        ----------
        inputs : dictionary containing names as keys and ndarrays as values
        output_keys : array of keys to use for the output ndarray dictionary
        dtype (optional) : type of the output datasets (otherwise same as one 
              of the input datasets.)
        """
        self._input_arrays = inputs
        shape = None
        
        
        for v in inputs.values() : 
            if shape == None : 
                shape = v.shape
            if shape != v.shape : 
                raise TypeError("All inputs must have the same shape.")
            if dtype == None : 
                dtype = v.dtype
        
        self._output_arrays = {}     
        for out_name in output_keys : 
            self._output_arrays[out_name]=np.empty(shape, dtype=dtype)
    
    def apply(self) : 
        """iterate over the array
        
        Note that the default behavior is to iterate over the array and 
        perform a calculation on each cell. If alternative behavior is desired,
        (such as, for instance, performing array math), this function should
        be overridden.
        """
        for i in np.ndenumerate(self._input_arrays[0]) : 
            self._apply_to_cell(i)
            
    def get_outputs(self) : 
        return self._output_arrays
        
    def get_inputs(self) : 
        return self._input_arrays
    
    @abstractmethod
    def _apply_to_cell(self, index) : 
        """given an index and the inputs, perform the operation and store result
        
        This method, implemented by the subclass, is how the inputs are mapped
        to the function/method arguments."""
        pass
    
class FunctionArrayOperator (CellArrayOperator) : 
    """Applies a function to individual cells of an array.
    
    In essense, this adapts a standalone function to the processing of 
    array data. Primarily, it maps the input ndarrays to function arguments, 
    and function output(s) to (the) output ndarray(s). To populate more than
    one output array, the registered function should return a tuple of 
    values.
    """
    def __init__(self, input_dict, output_keys, function) : 
        super(FunctionArrayOperator, self).__init__(input_dict, output_keys)
        self._function = function
        
    
    
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
        self._rr_view = self._history.view(NdarrayRoundRobin)
        self._rr_view.histdim = len(shape) # + 1 (history) - 1 (zero based)
        
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
    """ndarray subclass which implements round-robin indexing along one axis.
    
    A class which presents a view of an ndarray such that one of the axes behaves
    in a round robin fashion. The axis given round-robin properties is always 
    the length specified by the "shape" parameter, but the zero-point is incremented 
    whenever "next" is called. This means that index zero becomes the highest-index value
    and all other indices are effectively decremented by one. No data values are
    copied.
    
    This is never meant to be instantiated directly, only created as a view 
    of an existing ndarray.
    
    Element-wise operations will not assume round robin behavior, as these are
    typically implemented on the underlying databuffer and the databuffer is 
    not affected. The round robin behavior is exhibited on indexing only.
    
    Additionally, the round-robin functionality is very limited. Only explicit
    indices are allowed (no slicing, no ellipses, no array indexing). You are 
    pretty much limited to extracting or setting individual values.
    """
    def __array_finalize__(self, obj) :
        self.__i_history = 0
        self.histdim   = len(obj.shape) - 1
        self.obj = obj
        
    def next(self) : 
        self.__i_history = (self.__i_history + 1) % self.obj.shape[self.histdim]
        
    def get_history_index(self, hist) : 
        return (hist + self.__i_history) % self.obj.shape[self.histdim]
    
    def recalc_index(self, index) : 
        """Intercept and recompute the history dimension"""
        i_new = self.get_history_index(index[self.histdim])
        if self.histdim == 0 : 
            index = (i_new,) + index[1:]
        elif self.histdim == len(self.obj.shape)-1 :
            index = index[:-1] + (i_new,)
        else :
            index = index[:self.histdim] + (i_new,) + index[self.histdim+1:]
 
        return index
                
    def __getitem__(self, index) : 
        return self.obj[self.recalc_index(index)]
        
    def __setitem__(self, index, val) :
        self.obj[self.recalc_index(index)] = val