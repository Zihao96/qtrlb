import numpy as np
from scipy.linalg import sqrtm




TOMOGRAPHY_GATE_SETS = {
    'yurtalan': {'d': 3,
                 'gates': ['I',
                           'X180_01', 
                           'X90_01', 
                           'Y90_01', 
                           ('X180_01', 'X90_12'), 
                           ('X180_01', 'Y90_12'), 
                           ('X180_01', 'X90_12', 'X180_01'), 
                           ('X180_01', 'Y90_12', 'X180_01'), 
                           ('X180_01', 'X180_12', 'X180_01')]
                 },
}




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


def calculate_single_qudit_density_matrix(populations: np.ndarray, 
                                          tomography_gates_list: list[dict[str: list[str]]]) -> np.ndarray:
    """
    Give a measured population with shape (n_readout_levels, n_tomography_gates) and a gates list, \
    return a density matrix.
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
    measurement_opeators_tomo = [g.H @ m @ g for m in measurement_opeators_native for g in tomography_gates_values]

    density_matrix = reconstruct_dm_linreg(populations.flatten(), measurement_opeators_tomo, d)
    return density_matrix


def calculate_tomography_gates(tomography_gates_list: list[dict[str: list[str]]], d: int) -> list[np.ndarray]:
    """
    For a list of gate dict, calculate their actual matrix value.
    There won't be missing qudit or gate in each dict. Even Identity is explicit here.
    """
    tomography_gates_values = []

    for gate in tomography_gates_list:
        gate_strings = list(gate.values())  
        # Each element is a single gate string or a list of such string for single qudit.

        # A tensor-producted ndarray.
        gate_values = tensor_product_gates(gate_strings, d)
        tomography_gates_values.append(gate_values)

    return tomography_gates_values


def tensor_product_gates(gate_string: list, d: int) -> np.ndarray:
    """
    Give a list of string, return their tensor-producted value.
    Each element can either be string or a list of string.
    """
    return


def reconstruct_dm_linreg(result: np.ndarray, operators: list[np.ndarray], d: int):
    """
    Reconstruct density matrix based on measurement results and measurement operators.
    Ref: https://www.nature.com/articles/srep03496
    """
    return

