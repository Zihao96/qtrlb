# =============================================================================
# All the function in this script are supposed to be purely mathematical without
# considering the parameter or dictionary structure of measurement, so that it 
# could also be called for other purpose.
# 
# Support both nested list and ndarray. Please make sure the first index is 0
# for I_data and 1 for Q_data, which make the typical input_data has shape
# (2, n_reps, x_points).
# 
# The input_data = np.array(input_data) not only guarantee the data format,
# but also protect the original object and keep it unchanged.
#
# All functions should support both Scan and Scan2D where the data for 1D has 
# shape (2, n_reps, x_points) and Scan2D has shape (2, n_reps, y_points, x_points).
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
PI = np.pi


def rotate_IQ(input_data: list | np.ndarray, angle: float):
    """
    Rotate all IQ data with angle in radian.
    """
    input_data = np.array(input_data)
    if angle < -2*PI or angle > 2*PI:
        print(f'Processing: Rotate angle {angle} may not in radian!')
        
    rot_matrix = [[np.cos(angle), -np.sin(angle)], 
                  [np.sin(angle), np.cos(angle)]]
    
    result = np.einsum('ij,j...->i...', rot_matrix, input_data)
    return result


def autorotate_IQ(input_data: list | np.ndarray, n_components: int):
    """
    Automatically rotate all IQ data based on most distance Gaussian blob.
    """
    input_data = np.array(input_data)
    means, covariances = gmm_fit(input_data, n_components=n_components)
    point_i, point_j = find_most_distant_points(means)
    angle = -1 * np.arctan2(point_i[1]-point_j[1], point_i[0]-point_j[0])
    result = rotate_IQ(input_data, angle)
    return result
    

def gmm_predict(input_data, means, covariances, covariance_type='spherical', lowest_level: int = 0):
    """
    Predict the state of input data based on given means and covariances of GMM.
    By default, means should have shape (n_components, 2) for 2D gaussian.
    Covariances should have shape (n_components,) for symmetrical distribution,
    where n_components is the number of Gaussian blob in IQ plane.
    The return values are always count from zero.
    
    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """
    input_data = np.array(input_data)
    means = np.array(means)
    covariances = np.array(covariances)
    n_components = len(means)
    
    gmm = GaussianMixture(n_components, covariance_type=covariance_type)
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, covariance_type)
    gmm.weights_  = np.ones(n_components) / n_components

    result = lowest_level + gmm.predict(input_data.reshape(2,-1).T).reshape(input_data.shape[1:])
    # Magic reshape stealing from Ray.
    return result


def gmm_fit(input_data, n_components: int, covariance_type='spherical'):
    """
    Fit the input data with GMM. User must specify number of Gaussian blobs.
    The input_data should has shape (2, ...) because of two quadratures.
    Return the means and covariances. Means have shape (n_components, 2).
    Covariances have shape (n_components,) for symmetrical 2D distribution.

    Note from Zihao(11/07/2023):
    Gaussian Mixture doesn't support np.ma.core.MaskedArray.
    The mask will be dropped in gmm._validate_data().
    As the result, what ever mask we have will yield same gmm parameters.
    User must slice the data by themself before sending into this function.
    """
    assert not hasattr(input_data, 'mask'), 'Processing: MaskedArray are not supported by GaussianMixture.'
    input_data = np.array(input_data)
    gmm = GaussianMixture(n_components, covariance_type=covariance_type)
    gmm.fit(input_data.reshape(2,-1).T)
    return gmm.means_, gmm.covariances_


def heralding_test(*input_data: tuple[np.ndarray], trim: bool = True) -> np.ndarray:
    """
    Generate the ndarray mask with shape (n_reps, x_points).
    The input data should be arbitrary numbers of GMM predicted result with same data shape.
    The entries of mask will be 0 only if all input data is 0 at that position(index).
    It means for that specific repetition and x_point, all resonators pass heralding test.
    We then trim data to make sure all x_point has same amount of available repetition.
    Data trim doesn't Support 2D scan result.
    
    Note from Zihao(02/21/2023):
    The code here is stolen from original version of qtrl where we can only test ground state.
    However, ground state has most population and if our experiment need to start from |1>, pi pulse it.
    """
    mask = 0
    for data in input_data: mask = mask | (data != 0)
    mask = trim_mask(mask) if trim is True else mask
    return mask


def trim_mask(mask: list | np.ndarray) -> np.ndarray:
    """
    Given a mask with shape (n_reps, x_points) and only 0 and 1 as entries,
    return a mask where each column has same number of 1 by flip some 0 to 1.
    This will ensure that our repetition number is same for all x_points.

    Note from Zihao(11/08/2023):
    This is old qtrlb code. I didn't change the core algorithm.
    Please make it better if you know how to do it.
    heralding_test is not the only place to use this function!!!
    """
    mask = np.array(mask)
    assert len(mask.shape) == 2, 'Process: Do not support trim other than 2D data yet.'

    n_pass_min = np.min(np.sum(mask == 0, axis=0))  

    for i in range(mask.shape[1]):  # Loop over each x_point
        j = 0
        while np.sum(mask[:, i] == 0) > n_pass_min:
            n_short = np.sum(mask[:, i] == 0) - n_pass_min
            mask[j : j + n_short, i] = 1
            j += n_short
    return mask


def normalize_population(input_data, levels: list | np.ndarray, axis: int = 0, mask: np.ndarray = None):
    """
    Count population (specific interger) for different levels along a given axis.
    Return to normalized population (counts of appearing) with shape (n_levels, x_points).
    Allow a mask to pick entries in input_data to be normalized.
    Typically, the input_data and mask should have shape (n_reps, x_points).
    
    Example: 
        n_reps=4, x_points=3, two level system, no mask.
        data = [[0,1,0],[0,0,0],[0,1,0],[0,1,0]]
        result = [[1.0, 0.25, 1.0], [0.0, 0.75, 0.0]]
        So first x_point has 100% population in |0>, 0% in |1>.
        Second x_point has 25% population in |0>, 75% in |1>.
        Third x_point has 100% populaion in |0>, 0% in |1>.
    """
    # Zihao(02/17/2023): It's short, but still worth a function with clear explanation.
    masked_data = np.ma.MaskedArray(input_data, mask=mask)
    result = [np.mean(masked_data==level, axis=axis) for level in levels]
    return np.array(result)


def correct_population(input_data, corr_matrix: list | np.ndarray, corr_method: str = None):
    """
    Correct population based on a correction matrix (modification of confusion matrix).
    The element (row i, column j) in corr_matrix is P(predicted as state i | actually in state j).
    Thus, the corr_matrix times actual population gives predicted population.
    Least squares method works even if corr_matrix is not square matrix.

    Note from Zihao(09/12/2023):
    We must reshape the input data to two dimension, and the zero-th axis is different population.
    It's because np.linalg.solve only support this shape and more dimension cause ValueError.
    It's also because in least_squares, we need to manually loop over all other axis.
    When developing, keep mind to reshape the result back before return.
    """
    input_data = np.array(input_data)
    corr_matrix = np.array(corr_matrix)
    flat_data = input_data.reshape(input_data.shape[0], -1)

    # No correction.
    if corr_method is None:
        result = input_data

    # Inverse correction matrix.
    elif corr_method == 'pseudo_inverse':
        result = np.linalg.solve(corr_matrix, flat_data).reshape(input_data.shape)

    # Least squares minimization without bounds.
    elif corr_method == 'least_squares':
        # To support corr_matrix with arbitrary shape, I didn't use flat_data.shape.
        corrected_population = np.zeros((corr_matrix.shape[-1], flat_data.shape[-1]))

        for j in range(flat_data.shape[-1]):
            predicted_population = flat_data[:, j]  # Population vector for single x/y point.
            x0 = np.random.rand(corr_matrix.shape[-1])  # Unnormalized initial guess.

            corrected_population[:, j] = minimize(
                fun = lambda x: sum((np.dot(corr_matrix, x) - predicted_population) ** 2), 
                x0 = x0 / sum(x0), 
                method = "SLSQP", 
                constraints = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}, 
                # bounds=((0, 1) for _ in x0),  # Intentionally leaved commented.
                tol=1e-6
            ).x
        result = corrected_population.reshape((corr_matrix.shape[-1], *input_data.shape[1:]))

    else:
        raise ValueError(f'correct_population: corr_method {corr_method} is not supported.')
    
    return result


def find_most_distant_points(input_data):
    """
    Find the most distant points of data based on Euclidean distance.
    The input_data should has shape (n_points, n_dimension).
    
    Note from Zihao(02/20/2023):
        It's O(N^2) now. I know there is better way to do that. 
        Please do it if you know how.
    """
    input_data = np.array(input_data)  
    max_distance = 0
    for i in input_data:
        for j in input_data[i:]:
            distance_ij = np.linalg.norm(i-j)
            if distance_ij > max_distance:
                max_distance = distance_ij
                point_i = i 
                point_j = j
    return point_i, point_j


def get_readout_fidelity(confusion_matrix: list | np.ndarray) -> float:
    """
    Calculate readout fidelity based on a given confusion matrix.
    We are using form of confusion matrix such that sum vertical elements give 1.
    
    Reference:
    https://www.nature.com/articles/s41534-023-00689-6
    Page 3, definition of F_a before Eq.(4)

    Note from Zihao(05/22/2023):
        The equation above is related to its context in paper.
        I then read carefully about how they actually do three level fidelity.
        It's really just mean of diagonal. You can try Fig.4(b) about 96.9%.
    """
    fidelity = np.mean(np.diagonal(confusion_matrix))
    return float(fidelity)


def plot_corr_matrix(corr_matrix: list | np.ndarray) -> plt.Figure:
    """
    A pretty way to visualize correction matrix.
    """
    corr_matrix = np.array(corr_matrix)
    is_square = (corr_matrix.shape[0] == corr_matrix.shape[1])

    if is_square is True:
        title = f'{len(corr_matrix)}-state Readout Fidelity: {get_readout_fidelity(corr_matrix) * 100: 0.3g}%'
    else:
        title = 'Readout Correction Matrix'

    # Plot matrix in shaded orange color.
    fig, ax = plt.subplots(1, 1, figsize=(corr_matrix.shape[1], corr_matrix.shape[0]), dpi=400)
    ax.matshow(corr_matrix, cmap=plt.cm.Oranges)
    ax.xaxis.set_ticks_position('bottom')
    ax.set(xlabel='Prepared State', ylabel='Assigned State', title=title)

    # Add value as text to that grid.
    for row in range(corr_matrix.shape[0]):
        for col in range(corr_matrix.shape[1]):
            value = corr_matrix[row, col]
            text, fontsize = (f'{value :0.1e}', 6) if value < 0.01 else (f'{value :0.2g}', 10)
            ax.text(col, row, text, va='center', ha='center', fontsize=fontsize)
    return fig


def two_tone_predict(input_data_0: list | np.ndarray, 
                     input_data_1: list | np.ndarray, 
                     levels_0: list | np.ndarray, 
                     levels_1: list | np.ndarray) -> tuple[np.ndarray]:
    """
    Classify the qudit state based on result of GMM prediction from two tones.
    Levels are list of possible state assignment result of corresponding tones.
    We need one and only one element appearing in both levels list.
    We will generate a mask to drop the data that has contradiction when normalizing it.
    The result and mask should have same shape as two input data.
    """
    input_data_0 = np.array(input_data_0) 
    input_data_1 = np.array(input_data_1)
    levels_0 = np.array(levels_0)
    levels_1 = np.array(levels_1)

    # Find intersection and check it's unique.
    intersection = np.intersect1d(levels_0, levels_1)
    assert len(intersection) == 1, 'More than one state are reading out by both tone!'
    intersection_array = intersection * np.ones(shape=input_data_0.shape)

    # Element-wise comparision for finding the contradiction.
    # 1-True is zero and will be kept, 1-False is one and will be masked.
    mask = 1 - (np.equal(intersection_array, input_data_0) | np.equal(intersection_array, input_data_1))  

    # For kept data, one of the two has to be the intersection.
    # Then the other one is what we need (get it by substraction).
    # For masked data, it doesn't matter :)
    result = input_data_0 + input_data_1 - intersection_array
    return result, mask


def two_tone_normalize(input_data_0: list | np.ndarray, 
                       input_data_1: list | np.ndarray, 
                       levels_0: list | np.ndarray, 
                       levels_1: list | np.ndarray,
                       axis: int = 0, 
                       mask: np.ndarray = None) -> np.ndarray:
    """
    Count population (a pair of interger) based on all possible outcome pairs along a given axis.
    Return to normalized population (counts of appearing) with shape (n_possible_pairs, x_points).
    Allow a mask to pick entries in input_data to be normalized.
    Require levels_0 to be smaller than levels_1 and only one intersection state.

    Example: levels_0 = [0, 1, 2, 3], levels_1 = [3, 4, 5, 6]
    There will be 16 different outcome (0-15), and we map them to: 1 * (x1 - 3) + 4 * x0
    The returned result will have shape (16, x_points)
    The later corr_matrix will be expected to have shape (16, 7)
    """
    
    input_data_0 = np.array(input_data_0) 
    input_data_1 = np.array(input_data_1)
    levels_0 = np.array(levels_0)
    levels_1 = np.array(levels_1)

    # Find intersection and check it's unique.
    intersection = np.intersect1d(levels_0, levels_1)
    assert len(intersection) == 1, 'More than one state are reading out by both tone!'

    flatten_data = (input_data_1 - intersection) + len(levels_1) * input_data_0
    levels = np.arange(len(levels_0) * len(levels_1))
    return normalize_population(flatten_data, levels, axis, mask)


def multitone_predict_sequential(*data_levels_tuple: tuple) -> np.ndarray:
    """
    Classify the single qudit state based on result of GMM prediction from multitones.
    The arguments (data_levels_args) passed in here should be tuples.
    The number of arguments should equal to the number of tones.
    Each tuples has two element where the first one is the GMM_predicted data (ndarray) of that tone.
    The second element is list of all possible readout assignment levels of corresponding tone.
    We required the arguments passed in here are in ascdending order.
    The first tones should readout lowest level and the last tone should readout highest level. 
    We need one and only one overlapped element appear in two neighbor levels list.

    Example of usage:
    dataA = [[1, 3], [0, 3]]
    leakA = [[0, 1], [0, 1]]
    flipA = [[1, 0], [1, 0]]

    dataB = [[3, 4], [6, 6]]
    leakB = [[0, 0], [1, 1]]
    flipB = [[1, 1], [0, 0]]

    dataC = [[6, 7], [7, 8]]

    result = multitone_predict_sequential((dataA, [0,1,2,3]), (dataB, [3,4,5,6]), (dataC, [6,7,8]))
    result == [[1, 4], [0, 8]]

    About this algorithm:
    In this method, I perfer to think about it as a water leaking from top layer down to lower layer.
    Here leak represent which data/position this layer want to leak to next layer.
    Hence (1-leak) means which data/position this layer are able to kept.
    The accumulated_leak counts all data/position leaking from all previous layers.
    Thus, (1-leak) * accumulated_leak are the data we catched from last layer and will kept this layer.
    """
    accumulated_leak = 1
    result = 0
    for (data_0, levels_0), (data_1, levels_1) in zip(data_levels_tuple[:-1], data_levels_tuple[1:]):

        # Find intersection and check it's unique.
        intersection = np.intersect1d(levels_0, levels_1)
        assert len(intersection) == 1, 'More than one state are reading out by two neighbor tones!'

        # Element of the leak where data_0 equal to intersection will be 1 (leak), else 0 (keep).
        leak = (data_0 == intersection).astype(int)
        result += data_0 * (1 - leak) * accumulated_leak
        accumulated_leak *= leak

    # data_1 will be kept as last input data even after for loop finished.
    # The iterator reaches end, but these local variables in function frame get kept.
    result += data_1 * accumulated_leak
    return result


def multitone_predict_mask(*data_levels_tuple: tuple) -> tuple[np.ndarray]:
    """
    Classify the single qudit state based on result of GMM prediction from multitones.
    For requirement of the arguments and usage, please refer to multitone_predict_sequential.
    This function generate exactly same classification result with a mask showing contradiction.

    About this algorithm:
    For each tone, we now need to know the position of upward leak and downward leak.
    For a given tone, there is two case.
    First, if itself doesn't leak, we require the tone below it to leak up and tone above it to leak down.
    Second, the tone leak. Then we are good, the mask at other layer will worry about it.
    Otherwise, there is contradiction, we will generate such mask of this layer.
    The returned mask is the OR operation among all these masks.
    """
    result = multitone_predict_sequential(*data_levels_tuple)

    # Correctness are protected by multitone_predict_sequential.
    intersections = [levels[-1] for _, levels in data_levels_tuple[:-1]]

    # Create leak list with length equal to n_tones.
    leak_low = ([np.zeros_like(result)] 
                + [data == intersections[i] 
                   for i, (data, _) in enumerate(data_levels_tuple[1:])])
    leak_high = ([data == intersections[i] 
                  for i, (data, _) in enumerate(data_levels_tuple[:-1])] 
                 + [np.zeros_like(result)])

    # In this mask, problematic data will be labeled as 1, others as 0.
    mask = 0
    for i in range(len(data_levels_tuple)):
        leak_others = [h for h in leak_high[:i]] + [l for l in leak_low[i+1:]]
        leak_prod = np.prod(leak_others, axis=0)
        mask = mask | 1 - (leak_low[i] | leak_high[i] | leak_prod) 

    return result, mask


def multitone_normalize(*data_levels_tuple: tuple, axis: int = 0, mask: np.ndarray = None) -> np.ndarray:
    """
    Count population (a pair of interger) based on all possible outcome pairs along a given axis.
    Return to normalized population (counts of appearing) with shape (n_possible_pairs, x_points).
    Allow a mask to pick entries in input_data to be normalized.
    Require levels_0 to be smaller than levels_1 and only one intersection state.

    Example: levels_0 = [0, 1, 2, 3], levels_1 = [3, 4, 5, 6]
    There will be 16 different outcome (0-15), and we map them to: 1 * x0 + len(levels_0) * (x1 - 3)
    The returned result will have shape (16, x_points)
    The later corr_matrix will be expected to have shape (16, 7)
    """
    accumulated_length = 1
    flatten_data = data_levels_tuple[0][0] * accumulated_length  # Nothing but initialize with lowest levels.

    for (data_0, levels_0), (data_1, levels_1) in zip(data_levels_tuple[:-1], data_levels_tuple[1:]):
        
        # Find intersection and check it's unique.
        intersection = np.intersect1d(levels_0, levels_1)
        assert len(intersection) == 1, 'More than one state are reading out by two neighbor tones!'

        accumulated_length *= len(levels_0)
        flatten_data += accumulated_length * (data_1 - intersection)

    levels = np.arange(accumulated_length * len(levels_1))
    return normalize_population(flatten_data, levels, axis, mask)


