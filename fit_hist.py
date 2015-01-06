import accum_hist as ah
import numpy as np
import statsmodels.api as sm
import netCDF4 as nc
import pandas as pd
from random import seed, random


def bin_center(edges) : 
    return (edges[1:]+edges[:-1]) /2  
    
class Dummy (object): 
    """Mimics a result object"""
    def conf_int(self) : 
        return self.conf  
    
class FunctionalFitForm (object) : 
    def __init__(self, data=None, centers=None, fit_params=None) :
        """common initialization logic for children
        
        Users may provide an array of data and bin centers (in which case
        the fit will be performed), or may provide precomputed fit_params
        in the form supplied by get_fitinfo() (in which case the object will
        just be loaded with the fit data.)
        
        If you do neither, the object will not be ready for use until you
        call fit() or load(). 
        """
         
        if not ((data is None) or (centers is None)):
            self.fit(data,centers)
        elif fit_params is not None : 
            self.load(fit_params) 

class ExpTauForm (FunctionalFitForm) : 
    fit_column_names = ['r_squared',
            'tau',
            'tau_stderr',
            'b',
            'b_stderr',
            'tau_pvalue',
            'b_pvalue',
            'tau_conf_low',
            'tau_conf_high',
            'b_conf_low',
            'b_conf_high' ]

         
    def fit(self,data, centers) : 
        i_good = np.where(data > 0)
        y = np.log(data[i_good])
        x = centers[i_good]
        X = sm.add_constant(x)
        self.results = sm.OLS(y,X).fit()
        self.params = {}
        self.params['b'] = self.results.params[0]
        self.params['tau'] = self.results.params[1]
    
    def inverse(self,y) : 
        """Evaluate the inverse function given the fit parameters"""
        return (np.log(y) - self.params['b'])/self.params['tau']
        
    def evaluate(self,x) :
        """Evaluate the forward function given the fit parameters""" 
        return np.exp(self.params['tau']*x + self.params['b'])
    
    
    def get_fitinfo(self) : 
        """Outputs a single row of numeric data containing the parameters and fit info
    
        Columns names are listed in fit_column_names.
        """
        retval = np.empty( (len(ExpTauForm.fit_column_names),), dtype=np.float64)
        retval[0] = self.results.rsquared
        retval[1] = self.params['tau']
        retval[2] = self.results.bse[1]
        retval[3] = self.params['b']
        retval[4] = self.results.bse[0]
        retval[5] = self.results.pvalues[1]
        retval[6] = self.results.pvalues[0]
        conf = self.results.conf_int()
        retval[7:9] = conf[1]
        retval[9:11] = conf[0]
        return retval

    def load(self, fit_data) : 
        """loads a fit from a numerical array.
        
        Array should have the same columns in the same order as produced by
        get_fitinfo.
        """
        self.params = {}
        self.params['tau'] = fit_data[1]
        self.params['b']   = fit_data[3]
        
        self.results = Dummy()
        self.results.rsquared = fit_data[0]
        self.results.bse = [ fit_data[4], fit_data[2] ] 
        self.results.pvalues = fit_data[6:4:-1]
        self.results.conf = [ fit_data[9:11], fit_data[7:9] ]

class PowerNForm (FunctionalFitForm) :  
    
    fit_column_names =  [
        'r_squared',
        'n',
        'n_stderr',
        'b',
        'b_stderr',
        'a',
        'a_prime',
        'inv_n',
        'n_pvalue',
        'b_pvalue',
        'n_conf_low',
        'n_conf_high',
        'b_conf_low',
        'b_conf_high'
    ] 
      
    def fit(self, data, centers) : 
        """Fits this functional form to the provided data and centers"""
        i_good = np.where(data>0)
        y = np.log(data[i_good])
        x = np.log(centers[i_good])
        X = sm.add_constant(x)
        self.results = sm.OLS(y,X).fit()
        
        self.params = {}
        self.params['b'] = self.results.params[0]
        self.params['n'] = self.results.params[1]
        self.params['a'] = np.exp(self.params['b'])
        self.params['a_prime'] = np.exp(-self.params['b']/self.params['n'])
        self.params['inv_n'] = 1./self.params['n']
    

    
    def inverse(self,y) : 
        """Evaluate the inverse function given the fit parameters"""
        return self.params['a_prime'] * np.power(y,self.params['inv_n'])
        
    def evaluate(self, x):
        """Evaluate the forward function given the fit parameters""" 
        return self.params['a'] * np.power(x, self.params['n'])
    
    def get_fitinfo(self) : 
        """Outputs a single row of numeric data containing the parameters and fit info
    
        Columns names are listed in fit_column_names.
        """
        retval = np.empty( (len(PowerNForm.fit_column_names),), dtype=np.float64)
        retval[0] = self.results.rsquared
        retval[1] = self.params['n']
        retval[2] = self.results.bse[1]
        retval[3] = self.params['b']
        retval[4] = self.results.bse[0]
        retval[5] = self.params['a']
        retval[6] = self.params['a_prime']
        retval[7] = self.params['inv_n']
        retval[8:10] = self.results.pvalues[::-1]
        conf = self.results.conf_int()
        retval[10:12] = conf[1]
        retval[12:] = conf[0]
        return retval
        
    def load(self, fit_data) : 
        """loads a fit from a numerical array.
        
        Array should have the same columns in the same order as produced by
        get_fitinfo.
        """
        self.params = {}
        self.params['n'] = fit_data[1]
        self.params['b'] = fit_data[3]
        self.params['a'] = fit_data[5]
        self.params['a_prime'] = fit_data[6]
        self.params['inv_n'] = fit_data[7]
        
        self.results = Dummy()
        self.results.rsquared = fit_data[0]
        self.results.bse = [fit_data[4], fit_data[2]]
        self.results.pvalues = fit_data[9:7:-1]
        self.results.conf = [ fit_data[12:], fit_data[10:12] ]
        
        

forms = {} 
forms['exp_tau'] = ExpTauForm
forms['power_n'] = PowerNForm


class SparseHistoFit (object) : 
    """probabilities derived from histogram
    
    Initialized from a sparse histogram, this class fits the histogram from
    each bin with a functional form. The user may then draw from a probability 
    distribution based on the histogram.
    """
    def __init__(self, histo=None, ff_name=None, ff_dict=forms, weighted=False, min_npts=10) : 
        """specify the histogram and functional form to use for initialization, or none"""
        self.fits = {}
        self.default_fit = None
        self.ff_name = ff_name
        
        if ff_name is not None : 
            self._init_ff(ff_dict[ff_name])
            
        if histo is not None : 
            self.minmax = histo.minmax
            self._index = ah.init_indexers(self.minmax)
            self.default_contrib = histo.default_contrib
            
            self._fit_functional_form(histo, weighted, min_npts)
            
            seed(self)
        else: 
            self.default_contrib = {}

    def _init_ff(self, ff) : 
        self._fit_class = ff
        
    def _fit_functional_form(self, histo, weighted, min_npts) : 
        if weighted : 
            default = histo.default_weighted
        else : 
            default = histo.default
            
        #initialize parameters for all the bins.
        for i_combo in histo.get_combos(units=False) : 
            cur_hist = histo.get_histogram(i_combo, weighted=weighted, units=False)
            if np.count_nonzero(cur_hist) >= min_npts : 
                H_fit = self._fit_class(histo.get_histogram(i_combo, weighted=weighted, units=False), 
                           bin_center(histo.get_edges(i_combo, units=False)))
                self.fits[i_combo] = H_fit
                           
                        
        self.default_fit = self._fit_class( default, 
                        bin_center(histo.default_edges ))
                 
    def draw(self, combo, n=1) :
        """computes one or more histogram bin value(s) by inverting the fit"""
        i_combo = self._index.get_index(combo)
        if n == 1 : 
            x = random()
        else : 
            x = np.empty( (n,), dtype=np.float64)
            for i in range(n) : 
                x[i] = random()
        if i_combo in self.params : 
            fit = self.fits[i_combo]
            vals = fit.inverse(x)
        elif i_combo in self.default_contrib : 
            vals = self.default_fit.inverse(x)
        else : 
            if n==1:
                vals=0
            else :
                vals = np.zeros( (n,) ) 
        return vals
    
    def get_fit_cols(self) : 
        return self._fit_class.fit_column_names
            
    def get_fit_matrix(self) : 
        """returns the fit parameters as a pandas data frame"""
        colnames = self._fit_class.fit_column_names
        nrows = len(self.fits) + 1
        matrix = np.empty( (nrows, len(colnames)), dtype=np.float64)
        keys =  []
        
        # first row is default fit
        matrix[0,:] = self.default_fit.get_fitinfo()
        keys.append('default')
        
        # loop through all the fits
        i_row = 1
        for k,f in self.fits.iteritems() : 
            keys.append(k)
            matrix[i_row, :] = f.get_fitinfo()
            i_row += 1
            
        return pd.DataFrame(matrix, columns=colnames, index=keys)
        
    def load_fit_matrix(self, matrix, keys=None) : 
        """Loads fits from the given matrix.
        
        If the matrix is a pandas DataFrame, the tuples representing the 
        bin coordinates are expected to be provided as the index. If the 
        matrix is a plain ndarray, then the bin coordinates must be provided 
        in the keys parameter. In either case, the first row is considered to 
        be the default fit and the remainder have coordinates because they are
        tied to a specific bin.
        """
        if keys is None : 
            keys = matrix.index
            
        # first row is default fit
        self.default_fit = self._fit_class()
        self.default_fit.load(matrix[0,:])
        
        # loop through all the fits
        self.fits = {} 
        for i_row in range(1,matrix.shape[0]) : 
            f = self._fit_class()
            f.load(matrix[i_row,:])
            self.fits[ keys[i_row] ] = f               

def save_sparse_fit(fname, sfit) : 
    """saves sfit to a netCDF file "fname" """
    outfile = nc.Dataset(fname, 'w')
    
    fit_matrix = sfit.get_fit_matrix() 
    num_coords = len(fit_matrix.index[1])
    
    outfile.createDimension('fit_parameters', fit_matrix.shape[1])
    outfile.createDimension('fits', fit_matrix.shape[0])
    outfile.createDimension('ordinates', num_coords)
    
    # record the fit type
    outfile.fit_form = sfit.ff_name
    
    # record the column names for the fit_parameters table
    of_cols = outfile.createVariable('fit_parameters', str, ('fit_parameters',))
    of_cols[:] = sfit.get_fit_cols()
    
    # create variables 
    of_coords = outfile.createVariable('coordinates', dtype=np.int32, dims=('fits','ordinates'))
    of_params = outfile.createVariable('parameters', dtype=np.float64, dims=('fits','fit_parameters'))
    
    # fill the fit parameter matrix
    of_params[:] = fit_matrix
    
    # populate the coordinates
    for i_row in range(fit_matrix.shape[0]) : 
        if i_row == 0 :  
            of_coords[0,:] = np.ones( (num_coords,), dtype=np.int32 )  * -1
        else : 
            of_coords[i_row,:] = fit_matrix.index[i_row]
            
    outfile.close()
    
def load_sparse_fits(fname) : 
    """reads in a sparse fits object from a netCDF file"""
    infile = nc.Dataset(fname) 
    
    fit_matrix = infile.variables['parameters'][:]
    ff_name = infile.fit_form
    
    keys = [ 'default' ]
    for i_row in range(1,fit_matrix.shape[0]) : 
        keys.append( tuple(infile.variables['coordinates'][i_row,:]) )
            
    sfits = SparseHistoFit(ff_name=ff_name)
    sfits.load_fit_matrix(fit_matrix, keys=keys)
    return sfits
    
