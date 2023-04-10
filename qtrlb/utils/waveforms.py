import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sin
PI = np.pi




def get_waveform(length: int, shape: str, **waveform_kwargs):
    """
    Generate the waveform for Qblox to play. Input length should be integer in [ns].
    It has to return list, not ndarray.
    Each element has to be float (not integer) between -1.0 and 1.0.
    
    Note from Zihao(03/14/2023):
    When we design a waveform, it should better:
        1. have fully utilize 0.0--1.0 since we can change gain of each path later.
        2. have first/last data point as close to zero as possible.
        3. avoid steep change, especially on main waveform (derivative is fine).
    """
    
    # Check the length is integer
    if int(length) != length:
        print(f'The waveform length {length} is not interger and will be rounded to {round(length)}.')
        length = round(length)
        
    waveform = waveform_dict[shape](length, **waveform_kwargs).tolist()    
    return waveform
    

def plot_waveform(length: int, shape: str, **waveform_kwargs):
    """
    Plot waveform amplitude as time.
    """
    time = np.arange(length)
    waveform = get_waveform(length, shape)

    fig, ax = plt.subplots(1, 1, dpi=150)
    ax.plot(time, waveform)
    ax.set(xlabel='Time[ns]', ylabel='Amplitude[a.u.]', title=f'{length}ns {shape} waveform.',
           ylim=(-1.05, 1.05))
    for y in [-1, 0, 1]:
        ax.hlines(y=y, xmin=time[0], xmax=time[-1], color='black', linestyle='dashed', alpha=0.4)
    fig.show()    
    
    
def square(length: int, **waveform_kwargs):
    return np.ones(length, dtype=float)


def square_derivative(length: int, **waveform_kwargs):
    return np.zeros(length, dtype=float)
    
    
def gaussian(length: int, std: float = 0.15, **waveform_kwargs):
    """
    Ref: Eq.(20) in https://doi.org/10.1103/PhysRevA.96.022330
    Here std is a ratio and T is almost total length (which is half in Ref).
    T need to -1 since I want to make me first and last data point always be zero.
    """
    T = length - 1
    sigma = std * T
    t = np.arange(length)
    
    waveform = exp(-(t - T/2)**2 / 2 / sigma**2) - exp(-T**2 / 8 / sigma**2) 
    return waveform / np.max(np.abs(waveform))


def gaussian_derivative(length: int, std: float = 0.15, **waveform_kwargs):
    """
    Be careful about std here since large std will make first/last point far from zero.
    """
    T = length - 1
    sigma = std * T
    t = np.arange(length)
    
    waveform = - exp(-(t - T/2)**2 / 2 / sigma**2) * (t - T/2)  
    return waveform / np.max(np.abs(waveform))
    

def cos_square(length: int, ramp_fraction: float = 0.25, **waveform_kwargs):
    ramp_length = int(length * ramp_fraction)
    ramp_up = sin(PI / 2 * np.arange(ramp_length) / ramp_length) ** 2
    ramp_down = np.flip(ramp_up)
    return np.concatenate((ramp_up, np.ones(length-ramp_length*2, dtype=float), ramp_down))


def cos_square_derivative(length: int, ramp_fraction: float = 0.25, **waveform_kwargs):
    ramp_length = int(length * ramp_fraction)
    ramp_up = sin(PI * np.arange(ramp_length) / ramp_length)
    ramp_down = -1 * np.flip(ramp_up)
    return np.concatenate((ramp_up, np.zeros(length-ramp_length*2, dtype=float), ramp_down))
    

# Note from Zihao(03/24/2023): I don't like it at all, but:
# https://docs.python.org/3/faq/programming.html#how-do-i-use-strings-to-call-functions-methods
waveform_dict = {'square': square,
                 'square_derivative': square_derivative,
                 'gaussian': gaussian,
                 'gaussian_derivative': gaussian_derivative,
                 'cos_square': cos_square,
                 'cos_square_derivative':cos_square_derivative}    