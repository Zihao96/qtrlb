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


def fit(input_data: list | np.ndarray, x: list | np.ndarray, fitmodel: Model):
    """
    Fit data based on a given mathematical model.
    User can choose Model in this file or built-in lmfit Model.
    Return to a ModelResult object.
    Fitting result can be accessed by result.best_values (dict).
    The corresponding y-datapoint can be accessed by result.best_fit (ndarray).
    """
    input_data = np.array(input_data)
    x = np.array(x)
    fitmodel = fitmodel()
    params = fitmodel.guess(input_data, x)
    result = fitmodel.fit(input_data, params=params, x=x)
    return result


def exp_sin_func(x, tau, freq, phase, A, C):
    return C + A * exp(-x/tau) * sin(2*PI*freq*x + phase)

def exp_func(x, tau, A, C):
    return C + A * exp(-x/tau)

def quad_func(x, x0, A, C):
    return C + A * (x - x0)**2

def sin_func(x, freq, phase, A, C):
    return C + A * sin(2*PI*freq*x + phase)


class ExpSinModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(func=exp_sin_func, *args, **kwargs)
        
    def guess(self, data, x):
        sin_model = SineModel()
        sin_params = sin_model.guess(data, x=x)
        
        self.set_param_hint('freq', value=sin_params['frequency'].value/2/PI, min=0)
        self.set_param_hint('A', value=sin_params['amplitude'].value)
        self.set_param_hint('C', value=np.mean(data))
        self.set_param_hint('phase', value=sin_params['shift'].value, min=-np.pi, max=np.pi)
        self.set_param_hint('tau', value=(x[-1] - x[0])/2)
        return self.make_params()      


class ExpModel(Model):
    """Please do not use the built-in ExponentialModel of lmfit, because its guess is terrible.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(func=exp_func, *args, **kwargs)
        
    def guess(self, data, x):
        self.set_param_hint('A', value=np.max(data) - np.min(data))
        self.set_param_hint('C', value=np.mean(data))
        self.set_param_hint('tau', value=(x[-1] - x[0])/2, min=0)
        return self.make_params()   
    
    
class QuadModel(Model):
    """x0 is much more convenient than expression in  'a,b,c' form.
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
    """Consider the 2pi problem and add offset, comparing to SineModel.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(func=sin_func, *args, **kwargs)
        
    def guess(self, data, x):
        sin_model = SineModel()
        sin_params = sin_model.guess(data, x=x)
        
        self.set_param_hint('freq', value=sin_params['frequency'].value/2/PI, min=0)
        self.set_param_hint('A', value=sin_params['amplitude'].value)
        self.set_param_hint('C', value=np.mean(data))
        self.set_param_hint('phase', value=sin_params['shift'].value, min=-np.pi, max=np.pi)
        return self.make_params()  