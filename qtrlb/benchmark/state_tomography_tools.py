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