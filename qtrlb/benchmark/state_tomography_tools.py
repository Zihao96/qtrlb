import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.linalg import sqrtm
from itertools import combinations
from qtrlb.benchmark.RB1QB_tools import unitary
PI = np.pi




def get_simplest_gate_sets(d: int) -> list[list[str]]:
    """
    Return the simplest gate sets for single qudit state tomography of dimension d.
    We will map each off-diagonal element to diagonal by one of the gates in gate list.
    I choose them such that these sets will populate high level DOWN before measurement.

    Example: d = 4
    gate_list = [['I'],
                ['X90_01'],  # 01
                ['Y90_01'],
                ['X90_12'],  # 12
                ['Y90_12'],
                ['X90_23'],  # 23
                ['Y90_23'],
                ['X180_12', 'X90_01'],  # 02
                ['X180_12', 'Y90_01'],
                ['X180_23', 'X90_12'],  # 13
                ['X180_23', 'Y90_12'],
                ['X180_23', 'X180_12', 'X90_01'],  # 03
                ['X180_23', 'X180_12', 'Y90_01']] 
    """
    gate_list = [['I']]
    level_pairs = combinations([l for l in range(d)], 2)

    # Low and high will be two integers like (1, 4).
    for low, high in level_pairs:
        # ['X180_34', 'X180_23'] + ['X90_12']
        gate_x = [f'X180_{high-i-1}{high-i}' for i in range(high-low-1)]
        gate_x += [f'X90_{low}{low+1}']
        gate_list.append(gate_x)

        gate_y = [f'X180_{high-i-1}{high-i}' for i in range(high-low-1)]
        gate_y += [f'Y90_{low}{low+1}']
        gate_list.append(gate_y)

    return gate_list


TOMOGRAPHY_GATE_SETS = {
    # Ref: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.180504
    'Yurtalan': {
        'd': 3,
        'gates': [['I'],
                ['X180_01'], 
                ['X90_01'], 
                ['Y90_01'], 
                ['X90_12', 'X180_01'], 
                ['Y90_12', 'X180_01'], 
                ['X180_01', 'X90_12', 'X180_01'], 
                ['X180_01', 'Y90_12', 'X180_01'], 
                ['X180_01', 'X180_12', 'X180_01']]
    },
    # Ref: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.105.223601
    'Bianchetti': {
        'd': 3,
        'gates': [['I'],
                ['X90_01'],
                ['Y90_01'],
                ['X180_01'],
                ['X90_12'],
                ['Y90_12'],
                ['X180_01','X90_12'],
                ['X180_01','Y90_12'],
                ['X180_12','X180_01']]
    }
}

for i in range(3, 20):
    TOMOGRAPHY_GATE_SETS[f'Simplest_{i}'] = {'d': i, 'gates': get_simplest_gate_sets(i)}


################################################## 

def generalized_Gell_Mann_matrices(d: int) -> list[np.ndarray]:
    """
    Return the generalized Gell Mann matrices with dimension d.
    Ref: https://iopscience.iop.org/article/10.1088/1751-8113/41/23/235303
    See reference Eq.(3-5).

    Note from Zihao(09/15/23):
    I separate the 'for' loop not just for readibility.
    I found, surprisingly, combine them make it slower. 
    Test: 8.3s vs 10.6s for 1000 repetition with d = 49.
    I guess it's about how Jupyter in VSCode treat 'for' loop.
    """
    assert isinstance(d, int) and d >= 2, f'd must be integer no less than 2, not {d}'

    Gell_Mann_matrices = []

    # Adding symmetric GGM
    for k in range(1, d):
        for j in range(k):
            gmm = np.zeros((d, d), dtype=complex)
            gmm[j, k] = +1
            gmm[k, j] = +1
            Gell_Mann_matrices.append(gmm)

    # Adding antisymmetric GMM
    for k in range(1, d):
        for j in range(k):
            gmm = np.zeros((d, d), dtype=complex)
            gmm[j, k] = -1j
            gmm[k, j] = +1j
            Gell_Mann_matrices.append(gmm)

    # Adding diagonal GMM
    for l in range(1, d): 
        diagonals = [1 for _ in range(l)] + [-l] + [0 for _ in range(d - l - 1)]
        gmm = np.sqrt(2 / l / (l+1)) * np.diag(diagonals)
        Gell_Mann_matrices.append(gmm)

    return Gell_Mann_matrices


def state_fidelity(density_matrix: np.ndarray, ideal_density_matrix: np.ndarray) -> float:
    """
    Calculate state fidelity between two density matrix.
    Ref: Nielsen and Chuang Eq.(9.53)
    """
    fidelity = np.trace( sqrtm( 
        sqrtm(ideal_density_matrix) @ density_matrix @ sqrtm(ideal_density_matrix) 
    ) )
    return fidelity


def plot_density_matrix(density_matrix: np.ndarray, dpi=150) -> plt.Figure:
    """
    Plot the magnitude of each element in given density matrix.
    """
    matrix = np.abs(density_matrix)
    d = matrix.shape[0]

    fig = plt.figure(figsize=(d, d), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    x = [i for i in range(d) for _ in range(d)]  # The last 'for' generate inner layer. 
    y = [i for _ in range(d) for i in range(d)]
    z = np.zeros((d**2))
    dx = dy = 0.5 * np.ones((d**2))
    dz = matrix.flatten()

    ax.bar3d(x, y, z, dx, dy, dz)
    ax.set(xlabel='row', ylabel='column')
    return fig


def gate_str_to_matrix(gate_str: str, d: int = 2) -> np.ndarray:
    """
    Given a gate string and its dimension, return the matrix.
    Example: 'X180_12', d = 4, the result will be:
    [[1,  0,  0,  0],
     [0,  0, -1j, 0],
     [0, -1j, 0,  0],
     [0,  0,  0,  1]]
    """
    # Start from Identity
    matrix = np.eye(d, dtype=complex)
    if gate_str == 'I': return matrix

    try:
        gate, subspace = gate_str.split('_')
        l = int(subspace[0])  # The lower level in this subspace.
        axis, angle = gate[0], float(gate[1:]) / 180 * PI
    except ValueError:
        raise ValueError(f'STomo_tools: Cannot translate gate {gate_str} into matrix!')

    axis_dict = dict(zip(('X', 'Y', 'Z'), np.eye(3)))
    sub_matrix = unitary(angle, axis_dict[axis])
    matrix[l:l+2, l:l+2] = sub_matrix
    return matrix


def calculate_tomography_gates(tomography_gates_list: list[dict[str: list[str]]], d: int) -> list[np.ndarray]:
    """
    For a list of gate dict, calculate their actual tensor-producted matrix value.
    There won't be missing qudit or gate in each dict. Even Identity is explicit here.
    It supports multiple qudits.
    """
    tomography_gates_values = []

    # Loop over each gate dict in this gates list
    for gate_dict in tomography_gates_list:
        gate_arrays = []
        
        # Loop over each qudit in gate_dict
        for gate_list in gate_dict.values():
            matrix = np.eye(d)

            # Loop over each gate string in gate_list:
            for gate_str in gate_list:
                matrix = gate_str_to_matrix(gate_str, d) @ matrix

            gate_arrays.append(matrix)

        tensor_product_matrix = reduce(np.kron, gate_arrays)
        tomography_gates_values.append(tensor_product_matrix)

    return tomography_gates_values


def calculate_single_qudit_density_matrix(populations: np.ndarray, 
                                          tomography_gates_list: list[dict[str: list[str]]]) -> np.ndarray:
    """
    Calculate density matrix for given measured population and tomography gates list.
    The population should have shape (n_readout_levels, n_tomography_gates).
    It is not guaranteed that the returned matrix is positive semi-definite here.
    """
    d = populations.shape[0]

    # Several d * d matrix with only i on diagonal is 1 and others are 0. 
    measurement_opeators_native = [np.diag(row) for row in np.eye(d)]

    # A list of d * d ndarray, which are unitary matrices.
    # Each one corresponds to a gate dictionary in tomography_gates_list.
    tomography_gates_values = calculate_tomography_gates(tomography_gates_list, d)
    
    # Construct all measurement operators based on native operators and tomography gates.
    # The order here is C order which is default order of ndarray.flatten.
    # We will later flatten populations in same order.
    # Later comprehension is at inner layer. So loop g first, then m.
    measurement_opeators_tomo = [g.T.conj() @ m @ g 
                                 for m in measurement_opeators_native for g in tomography_gates_values]

    # ndarray.flatten() is deep copy.
    density_matrix = reconstruct_dm_linreg(populations.flatten(), measurement_opeators_tomo, d)
    return density_matrix


def reconstruct_dm_linreg(results: np.ndarray, operators: list[np.ndarray], d: int):
    """
    Reconstruct density matrix based on measurement results and measurement operators.
    Both of them follow same order where tomography gates inside different measurement outcome.
    Ref: https://www.nature.com/articles/srep03496

    Note from Zihao(09/18/2023):
    I intentionally use for loop instead of list comprehension / array operation.
    This is for readibility and debugging. 
    When it become performance bottleneck, I will change it.
    """
    Omegas = np.array(generalized_Gell_Mann_matrices(d))  # Shape (d**2 - 1, d, d)

    xTx = np.zeros((len(Omegas), len(Omegas)), dtype=complex)
    sum = np.zeros((len(Omegas), 1), dtype=complex)

    for i, m in enumerate(operators):
        # Equation below Eq.(2)
        psi = np.array([np.trace(m @ Omega_i) for Omega_i in Omegas]).reshape(len(Omegas), -1)
        # Equation below Eq.(8)
        xTx += psi * psi.T
        # Eq.(8), summation part.
        sum += psi * (results[i] - 1 / d)

    # Eq.(8)
    theta_ls = np.linalg.pinv(xTx, hermitian=True) @ sum

    # Eq.(1).
    dm = np.eye(d, dtype=complex) / d
    for i, theta in enumerate(theta_ls):
        dm += theta * Omegas[i]
    return dm


def make_dm_physical(dm: np.ndarray) -> np.ndarray:
    """
    Given a Hermitian density matrix, make it physical (positive and semi-definite).

    Ref: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.070502
    See Fast algorithm above Fig.(2)

    Note from Zihao(09/20/2023):
    This is implemented in original Berkeley code assuming Gaussian noise in Ref above.
    I cann't guarantee this is compatible with our linear regression algorithm.
    """
    dm /= np.trace(dm)

    # Step 1: Calculate eigenvalues and eigenvectors. Eigenvalues are in ascending order by default.
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals = eigvals[::-1]
    if eigvals[-1] >= 0: return dm

    # Step 2: Initialize lambda, i, a. 
    eigvals_new = np.zeros(eigvals.shape)
    i, a = len(eigvals), 0

    # Step 3: calculate a and i.
    while eigvals[i-1] + a / i < 0:
        a += eigvals[i-1]
        i -= 1

    # Step 4: calculate new eigenvalues.
    eigvals_new[:i] = eigvals[:i] + a / i

    # Step 5: construct new density matrix.
    eigvals_new = eigvals_new[::-1]
    dm_new = eigvecs @ np.diag(eigvals_new) @ eigvecs.T.conj()

    return dm_new