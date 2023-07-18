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

def gaussian1d_func(x, mean, std, A, C):
    return C + A * exp( -0.5 * ((x-mean) / std)**2 )

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


def resonator_hanger_transmission_func(x, f0, Q, Qc, theta, A, phi, ED, PCC):
    """
    Calculate S21 as a function of frequency.
    Here x is frequency, A and phi are global amplitude/phase factor.
    A and phi will effectively change the off-resonant amplitude/phase.
    ED is the electrical delay in unit of [s].
    PCC is Power Compression Coefficient to compensate natural power drop as mod_freq.
    PCC is in unit of [Amplitude/Hz], which is [mV/Hz] on Qblox.
    Reference:
    https://iopscience.iop.org/article/10.1088/2058-9565/ac070e
    """
    S21 =  1 - Q * exp(1j * theta) / Qc / (1 +  2j * Q * (x-f0) / f0)
    return A * exp(1j * (2*PI * (x-f0) * ED + phi)) * (1 + PCC * (x-f0)) * S21


def double_exp_sin_func(x, C_0, C_1, tau1, tau2R, A_0, A_1, freq_0, freq_1, phase_0, phase_1):
    """ 
    Ref: https://doi.org/10.1103/PhysRevLett.114.010501, Eq.(7) in Supplment Material.
    The function here is more general and can give better fit with T1 and separated amp.
    """
    return (C_0 + C_1 * exp(-x / tau1) 
            + exp(-x / tau2R) * (A_0 * sin(2*PI * freq_0 * x + phase_0) 
                                + A_1 * sin(2*PI * freq_1 * x + phase_1)))


def triple_exp_sin_func(x, C_0, C_1, tau1, tau2R, A_0, A_1, A_2, 
                        freq_0, freq_1, freq_2, phase_0, phase_1, phase_2):
    return (C_0 + C_1 * exp(-x / tau1) 
            + exp(-x / tau2R) * (A_0 * sin(2*PI * freq_0 * x + phase_0) 
                                + A_1 * sin(2*PI * freq_1 * x + phase_1)
                                + A_2 * sin(2*PI * freq_2 * x + phase_2)))


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
    

class ResonatorHangerTransmissionModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(func=resonator_hanger_transmission_func, *args, **kwargs)

    def guess(self, data, x):
        amplitude = np.abs(data)
        phase = np.angle(data)

        A_guess = np.max(amplitude)
        f0_guess = x[np.argmin(amplitude)]
        amplitude /= A_guess  # Normalize amplitude for later guess.

        # Guess all parameters except Qs.
        self.set_param_hint('f0', value=f0_guess, min=0)
        self.set_param_hint('theta', value=0, min=-PI, max=PI)
        self.set_param_hint('A', value=A_guess, min=0)
        self.set_param_hint('phi', value=np.mean(phase), min=-PI, max=PI)
        self.set_param_hint('ED', value=(phase[-1] - phase[0]) / (x[-1] - x[0]))
        self.set_param_hint('PCC', value=(amplitude[-1] - amplitude[0]) / (x[-1] - x[0]))

        # Guess Q.
        half_max = (np.max(amplitude) + np.min(amplitude)) / 2
        full_width = 2 * np.abs(f0_guess - x[np.argmin(np.abs(amplitude - half_max))])
        self.set_param_hint('Q', value=f0_guess/full_width, min=0)

        # Guess Qc.
        # Ref: https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html
        self.set_param_hint('Qc', value=f0_guess/full_width/(1-np.min(amplitude)), min=0)

        return self.make_params()  