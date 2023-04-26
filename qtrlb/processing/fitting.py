# =============================================================================
# All the function in this script are supposed to be purely mathematical without
# considering the parameter or dictionary structure of measurement, so that it 
# could also be called for other purpose.
# 
# Example: Suppose we have 'data' as y axis and a known x axis called 't'.   
# 
#   fitmodel = ExpSinModel()
#   params = fitmodel.guess(data, t)
#   result = fitmodel.fit(data, params=params, x=t)
#
# The fitting result is stored in a dictionary result.best_values.
# =============================================================================

import numpy as np
from lmfit import Model
from lmfit.models import SineModel, QuadraticModel
from numpy import exp, sin
PI = np.pi


def fit(input_data: list | np.ndarray, x: list | np.ndarray, fitmodel: Model, **fitting_kwargs):
    """
    Fit data based on a given mathematical model.
    User can choose Model in this file or built-in lmfit Model.
    Return to a ModelResult object.
    Fitting result can be accessed by result.best_values (dict).
    The corresponding fit-datapoint can be accessed by result.best_fit (ndarray).
    We allow keyword arguments here for 2D/3D fit and possible change.
    """
    input_data = np.array(input_data)
    x = np.array(x)
    fitmodel = fitmodel()
    params = fitmodel.guess(input_data, x, **fitting_kwargs)
    result = fitmodel.fit(input_data, params=params, x=x, **fitting_kwargs)
    return result


def exp_sin_func(x, tau, freq, phase, A, C):
    return C + A * exp(-x/tau) * sin(2*PI*freq*x + phase)

def exp_func(x, tau, A, C):
    return C + A * exp(-x/tau)

def quad_func(x, x0, A, C):
    return C + A * (x - x0)**2

def sin_func(x, freq, phase, A, C):
    return C + A * sin(2*PI*freq*x + phase)

def exp_func2(x, r, A, C):
    return C + A * (r ** x)


def chevron_func(x, y, omega_0, freq_offset, phase, A, C):
    """
    Ref: Gerry and Knight(2005) Eq.(4.80)
    Here x is the time/pulse_length, y is the detuning.
    """
    t, detuning = np.meshgrid(x, y)  # Return 2D data when t and detuning are both array.
    omega_R = np.sqrt( omega_0**2 + (detuning - freq_offset)**2 )
    return C + A * (omega_0/omega_R)**2 * sin(2*PI * omega_R * t / 2 + phase)**2


class ExpSinModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(func=exp_sin_func, *args, **kwargs)
        
    def guess(self, data, x):
        sin_model = SineModel()
        sin_params = sin_model.guess(data, x=x)
        
        self.set_param_hint('freq', value=sin_params['frequency'].value/2/PI, min=0)
        self.set_param_hint('A', value=sin_params['amplitude'].value)
        self.set_param_hint('C', value=np.mean(data))
        self.set_param_hint('phase', value=sin_params['shift'].value, min=0, max=2*PI)
        self.set_param_hint('tau', value=(x[-1] - x[0])/2, min=0)
        return self.make_params()      


class ExpModel(Model):
    """ Please do not use the built-in ExponentialModel of lmfit, because its guess is terrible.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(func=exp_func, *args, **kwargs)
        
    def guess(self, data, x):
        self.set_param_hint('A', value=np.max(data) - np.min(data))
        self.set_param_hint('C', value=np.mean(data))
        self.set_param_hint('tau', value=(x[-1] - x[0])/2, min=0)
        return self.make_params()   
    
    
class QuadModel(Model):
    """ x0 is much more convenient than expression in  'a,b,c' form.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(func=quad_func, *args, **kwargs)
        
    def guess(self, data, x):
        quad_model = QuadraticModel()
        quad_params = quad_model.guess(data, x=x)
        
        x0 = -quad_params['b']/quad_params['a']/2
        A = quad_params['a']
        C = quad_params['c'] - A * x0**2
        
        self.set_param_hint('x0', value=x0)
        self.set_param_hint('A', value=A)
        self.set_param_hint('C', value=C)
        return self.make_params() 
    
    
class SinModel(Model):
    """ Consider the 2pi problem and add offset, comparing to SineModel.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(func=sin_func, *args, **kwargs)
        
    def guess(self, data, x):
        sin_model = SineModel()
        sin_params = sin_model.guess(data, x=x)
        
        self.set_param_hint('freq', value=sin_params['frequency'].value/2/PI, min=0)
        self.set_param_hint('A', value=sin_params['amplitude'].value)
        self.set_param_hint('C', value=np.mean(data))
        self.set_param_hint('phase', value=sin_params['shift'].value, min=0, max=2*PI)
        return self.make_params()  
    
    
class ExpModel2(Model):
    """ A different exponential model where we fit base rather than life time.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(func=exp_func2, *args, **kwargs)
        
    def guess(self, data, x):
        self.set_param_hint('A', value=np.max(data) - np.min(data))
        self.set_param_hint('C', value=np.mean(data))
        self.set_param_hint('r', value=1, min=0)
        return self.make_params()
    

class ChevronModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(func=chevron_func, independent_vars=['x', 'y'], *args, **kwargs)
    
    def guess(self, data: np.ndarray, x: np.ndarray, **fitting_kwargs):
        self.set_param_hint('omega_0', value=2/(np.max(x)-np.min(x)))  # Assume two periods.
        self.set_param_hint('freq_offset', value=0)
        self.set_param_hint('phase', value=0)
        self.set_param_hint('A', value=np.max(data)-np.min(data))
        self.set_param_hint('C', value=np.mean(data))
        return self.make_params()