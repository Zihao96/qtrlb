import numpy as np
import matplotlib as mpl
import qtrlb.utils.units as u
import matplotlib.pyplot as plt 
from scipy.stats import norm
from qtrlb.processing.fitting import gaussian1d_func
from matplotlib.colors import LinearSegmentedColormap as LSC
PI = np.pi


################################################## 
# Color science

CIE_MATRIX = np.array([
    [-0.14282, 1.54924, -0.95641],
    [-0.32466, 1.57837, -0.73191],
    [-0.68202, 0.77073, 0.56332]
])


def sort_color_list_hue(color_list: list | np.ndarray) -> np.ndarray:
    """
    Sort the color list by the hue of the color.
    """
    color_list = np.array(color_list)
    hsv = mpl.colors.rgb_to_hsv(color_list[:, :3])
    return color_list[np.argsort(hsv[:, 0])]


def calculate_CCT(color: list | np.ndarray) -> float:
    """
    Calculate the Correlated Color Temperature (CCT) of the color.
    Ref: https://dsp.stackexchange.com/questions/8949/how-to-calculate-the-color-temperature-tint-of-the-colors-in-an-image
    """
    color = np.array(color)
    cie_value = np.matmul(CIE_MATRIX, color[:3])
    illmn = cie_value[1]
    x, y = cie_value[:2] / np.sum(cie_value)
    n = (x - 0.3320) / (0.1858 - y)
    cct = -449 * n**3 + 3525 * n**2 - 6823.3 * n + 5520.33
    return cct, illmn


def sort_color_list_CCT(color_list: list | np.ndarray) -> np.ndarray:
    """
    Sort the color list by the CCT and illuminance of the color.
    """
    color_list = np.array(color_list)
    ccts, illmns = np.zeros(len(color_list)), np.zeros(len(color_list))
    
    for l, color in enumerate(color_list):
        ccts[l], illmns[l] = calculate_CCT(color)
    return color_list[np.argsort(ccts)]


def plot_color_list(color_list: list | np.ndarray, dpi: int = 400):
    """
    Plot the color list in the color space.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 3.375), dpi=dpi)
    for l, color in enumerate(color_list):
        ax.plot((0, 1), (l, l), color=color)
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(color_list)), np.arange(len(color_list)), fontsize=6)
    ax.tick_params(axis='y', direction='in', length=2)
    return fig


def get_color_list(name: str = 'wzh') -> np.ndarray:
    """
    Get commonly used colors for plotting.
    """
    if name == 'wzh':
        # Sorted by the hue of the color
        color_list = np.concatenate((
            mpl.colors.to_rgba_array(plt.rcParams['axes.prop_cycle'].by_key()['color']),
            [mpl.colormaps['Set2'](i) for i in range(7)]
        ))
        color_list = sort_color_list_hue(color_list)
        color_list[[2,16]] = color_list[[16,2]]
        color_list[6] = color_list[6] * np.array([0.9, 0.9, 0.9, 1.0])
        color_list = np.delete(color_list, 14, axis=0)

    elif name == 'matplotlib':
        # Matplotlib default
        color_list = np.concatenate((
            mpl.colors.to_rgba_array(plt.rcParams['axes.prop_cycle'].by_key()['color']),
            [mpl.colormaps['Set2'](i) for i in range(7)]
        ))

    elif name == 'cQED':
        # color list used by Prof. Alexandre Blais's group
        color_list = mpl.colors.to_rgba_array(
            ["#2F4858", "#A02829", "#33658A", "#E6823B", "#D34E5B", "#4A8C82", 
             "#C0A184", "#86BBD8", "#6B92A4", "#CFB54E", "#97A169", "#C9A0DC"]
        )

    else:
        raise ValueError(f'The color list name {name} is not recognized.')

    return color_list


COLOR_LIST = get_color_list()




##################################################
# Plotting functions

def plot_IQ(ax: plt.Axes, I: np.ndarray, Q: np.ndarray, c: np.ndarray = None, **plot_setting):
    """
    Make scatter IQ plot with possible color and color map. Return the plt.Figure object.
    """
    if c is not None: 
        cmap = LSC.from_list('qtrlb', COLOR_LIST[c.min(): c.max()+1])
        c = (c - c.min()) / c.max()  # Normalize the color to [0, 1].
    else:
        cmap = None
    ax.scatter(I, Q, c=c, cmap=cmap, alpha=0.2)
    ax.axvline(color='k', ls='dashed')    
    ax.axhline(color='k', ls='dashed')
    ax.set(xlabel='I', ylabel='Q', aspect='equal', **plot_setting)
    return ax


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
