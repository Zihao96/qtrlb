# =============================================================================
# Note from Zihao (07/18/2023):
# This file includes some code that is hard to classified.
# There is also some code that is not ready for integration.
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
import qtrlb.utils.units as u
from scipy.stats import norm
from qtrlb.processing.fitting import gaussian1d_func
PI = np.pi




COLOR_LIST = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap_set3 = mpl.cm.get_cmap('Set3')
COLOR_LIST.extend([mpl.colors.rgb2hex(cmap_set3(i)) for i in range(12)])




def plot_overnightscan_result(total_tau_dict: dict, time_list: list | np.ndarray = None) -> dict[str: tuple]:
    """
    Plot the result of an Overnight coherence Scan.
    All coherence time will be in unit of [us].

    Note from Zihao (forget the exact time, 2022 summer):
    I decided to separate the plots by qubit so that we will have single plot sheet for each qubit.
    It will look like a datasheet of a electronics.
    How to separate plots should be determined by what we want to compare.
    I believe the different subspace definitely worth comparison.
    So there is only two more direction: compare scan type and separate qubits, or conversely.
    The reasons I choose to separate qubit are: 
        1. qubits properties vary as fabrication process fluctuation, so it's unfair to compare; 
        2. sometimes the subspace of qubits we have calibrated are not same with each other, but after calibration, \
            we can always do all type of scan, which make the plot uniform in shape.
    The real reason is that it's just easier and I'm lazy.
    """
    figs = {}
    Gaussian_norm = lambda x, mean, std: gaussian1d_func(x, mean, std, A=1/std/np.sqrt(2*PI), C=0)

    for qubit, qubit_dict in total_tau_dict.items():
        # Get the figure size
        nrow = len(qubit_dict)  # Number of subspace
        ncolumn = max([len(qubit_dict[subspace]) for subspace in qubit_dict])  # Max number of scans

        fig_hist, axes = plt.subplots(nrow, ncolumn, figsize=(8*ncolumn, 6*nrow), squeeze=False)  
        fig_time, axes2 = plt.subplots(nrow, ncolumn, figsize=(8*ncolumn, 6*nrow), squeeze=False)
        # Disable squeeze to avoid that single subplot will return to scalar which doesn't have subscript.

        for i, subspace in enumerate(qubit_dict):
            for j, scan in enumerate(qubit_dict[subspace]):
                tau_list = np.array(qubit_dict[subspace][scan]) / u.us 

                # Directly fit with scaled gaussian, 
                # Ref: https://stackoverflow.com/questions/7805552/fitting-a-histogram-with-python
                (mean, std) = norm.fit(tau_list)
                
                # Normalization 
                # Ref: https://stackoverflow.com/questions/3866520/plot-a-histogram-such-that-bar-heights-sum-to-1-probability 
                weights = np.ones_like(tau_list) / len(tau_list)
                
                # Plot the histogram. The default bins is 10, "normed" argument doesn't work here.
                n, bins, patches = axes[i][j].hist(tau_list, weights=weights, color='indigo', alpha=0.35, rwidth=0.8, zorder=0)   
            
                # Plot the Gaussian fit scatter
                area = 0
                for k in range(len(n)):
                    area += n[k] * (bins[k+1] - bins[k])
                fit_value = [Gaussian_norm(tau, mean, std) * area for tau in tau_list]
                axes[i][j].scatter(tau_list, fit_value, color='lightsteelblue', zorder=1)
            
                # Plot the Gaussian fit line
                x = np.linspace(min(tau_list), max(tau_list), 100)
                y = Gaussian_norm(x, mean, std) * area 
                axes[i][j].plot(x, y, color='lightsteelblue', linewidth=1.5, label=f'mean={mean:.3g} [us],\nstd={std:.3g} [us]')   
                axes[i][j].set(xlabel=f'{scan} [us]', ylabel='Probability', title=f'{scan}_{qubit}/{subspace}')
                axes[i][j].legend(loc='upper left')
                
                # Plot result in time domain
                if time_list is not None:
                    axes2[i][j].plot(time_list, tau_list)
                    axes2[i][j].set(xlabel='Time [hrs]', ylabel=f'{scan} [us]', title=f'{scan}_{qubit}/{subspace}')

        figs[qubit] = (fig_hist, fig_time)

    return figs

