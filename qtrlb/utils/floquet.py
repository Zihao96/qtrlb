import numpy as np
import qutip as qt
from tqdm.notebook import trange
from scipy.signal import find_peaks




def solve_Floquet_esys(
    hamiltonian: qt.QobjEvo, 
    T: float, 
    solver_options: dict = None
) -> np.linalg._linalg.EigResult:
    """
    Return the Floquet quasienergies and modes of a given periodic Hamiltonian.
    """    
    U = qt.solver.Propagator(hamiltonian, memoize=1, options=solver_options)
    U_T = U(T)
    return np.linalg.eig(U_T.data_as('ndarray'))


def solve_Floquet_esys_parallel(
    hamiltonians: list[qt.QobjEvo],
    T: float,
    solver_options: dict = None
) -> tuple[np.ndarray]:
    """
    Return arrays of Floquet quasienergies and modes.
    The hamiltonian is a list of QobjEvo objects.

    Notes:
    1. numpy.linalg.eig is 25% faster than qutip.core.data.eigs
    2. list.append() is 60% faster than assigning value into empty pre-constructed array.
    3. U_T = U(T) returns a Qobj. This line is equivalent to U.solver.step(T).
       It's the most time-consuming line, because it's integrating the differential equation.
    4. The number of CPU cores can be found by available_cpu_count() in qutip.settings module.
    """
    
    if solver_options is None:
        solver_options = {'method': 'adams', 'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 20000}

    results = qt.solver.parallel.parallel_map(
        task=solve_Floquet_esys,
        values=hamiltonians,
        task_kwargs={'T': T, 'solver_options': solver_options},
        progress_bar='tqdm'
    )
    quasienergies_array = []
    mode_mat_array = []
    for (evals, evecs) in results:
        quasienergies_array.append(-np.angle(evals) / T)
        mode_mat_array.append(evecs)

    return np.array(quasienergies_array), np.array(mode_mat_array)


def solve_Floquet_esys_serial(
    hamiltonian: callable,
    T: float, 
    amps: np.ndarray,
    solver_options: dict = None
) -> tuple[np.ndarray]:
    """
    Return arrays of Floquet quasienergies and modes.
    The hamiltonian is expected to be a callable that takes an amplitude as input 
    and returns a QobjEvo object.

    Notes:
    1. numpy.linalg.eig is 25% faster than qutip.core.data.eigs
    2. list.append() is 60% faster than assigning value into empty pre-constructed array.
    3. U_T = U(T) returns a Qobj. This line is equivalent to U.solver.step(T).
       It's the most time-consuming line, because it's integrating the differential equation.
    4. The number of CPU cores can be found by available_cpu_count() in qutip.settings module.
    """
    if amps[0] != 0:
        raise ValueError("The first amplitude must be 0.")
    if solver_options is None:
        solver_options = {'method': 'adams', 'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 20000}
    
    quasienergies_array = []
    mode_mat_array = []
    for i in trange(len(amps)):
        U = qt.solver.Propagator(hamiltonian(amps[i]), memoize=1, options=solver_options)
        U_T = U(T)
        evals, evecs = np.linalg.eig(U_T.data_as('ndarray'))
        quasienergies_array.append(-np.angle(evals) / T)
        mode_mat_array.append(evecs)

    return np.array(quasienergies_array), np.array(mode_mat_array)


def sort_Floquet_branches(
    quasienergies_array: np.ndarray,
    mode_mat_array: np.ndarray,
    number_operators: np.ndarray = None
) -> tuple[np.ndarray]:
    r"""
    Return the sorted arrays of Floquet quasienergies and modes.
    The quasienergies_array is expected to be of shape (n_amps, n_levels),
    the mode_mat_array is expected to be of shape (n_amps, n_levels, n_levels),
    and the number_operators is expected to be of shape (n_levels, n_levels).

    This function is useful when using Floquet branch analysis for analyzing 
    multilevel quantum systems with a periodic drive. When the amplitude of the drive
    is changed slowly compared to the drive frequency, the dynamics of the system can be 
    analyzed from the instantaneous Floquet spectrum. The Floquet quasienergies and modes
    at different amplitudes form a set of branches as they are sorted by the overlap
    between the Floquet modes of adjacent amplitudes.
    Ref: https://doi.org/10.1103/PhysRevX.14.041023

    The Floquet mode matrix at the i-th amplitude be expressed as:
        [\ket{phi^{(i)}_0}, \ket{phi^{(i)}_1}, ..., \ket{phi^{(i)}_{n_levels-1}}],
    where each column is a Floquet mode vector expressed in the bare-state basis.

    An example of the tracking_indices and indices_branches: 
        Assuming n_amps=5, n_levels=3,
        tracking_indices = [
            [0, 1, 2],  # Amp 0 to amp 1
            [1, 0, 2],  # Amp 1 to amp 2, indicate a branch switching event
            [0, 1, 2],  # Amp 2 to amp 3
            [1, 0, 2]   # Amp 3 to amp 4, indicate a branch switching event
        ]
        The corresponding indices_branches would be:
        indices_branches = [
            [0, 1, 2],  # Amp 0
            [0, 1, 2],  # Amp 1, no change
            [1, 0, 2],  # Amp 2, switch indices
            [1, 0, 2],  # Amp 3, no change
            [0, 1, 2]   # Amp 4, switch indices
        ]
    """
    
    n_amps, n_levels = quasienergies_array.shape
    mode_mat_array_H = mode_mat_array.transpose(0, 2, 1).conj()
    if number_operators is None: number_operators = np.arange(n_levels) * np.eye(n_levels)

    # Calculate the average bare-state photon number of all Floquet modes for all amplitudes.
    # Result has shape (n_amps, n_levels).
    avg_num_array = np.diagonal(mode_mat_array_H @ number_operators @ mode_mat_array, 0, -2, -1)
    avg_num_array = np.abs(avg_num_array)

    # Calculate the overlaps between Floquet modes matrices of two adjacent amplitudes i and i+1.
    # Result has shape (n_amps-1, n_levels, n_levels).
    # The i-th element in the first axis gives an overlap matrix with analytical expression:
    # \sum_{j, k} \braket{\phi^{(i)}_j|\phi^{(i+1)}_k} \ket{j}\bra{k}
    overlaps = np.abs(mode_mat_array_H[:-1] @ mode_mat_array[1:])

    # Use the maximum overlap to tracking the change of indices between adjacent amplitudes.
    # Result has shape (n_amps-1, n_levels).
    # The j-th element in the second axis is an integer value k 
    # meaning the k-th vector in the eigenvector matrix of the (i+1)-th amplitude
    # should be sorted into the same index of 
    # the j-th vector in the eigenvector matrix of the (i)-th amplitude.
    tracking_indices = np.argmax(overlaps, axis=-1)

    # Construct an indices matrix for sorting the Floquet quasienergies and modes.
    # The first row is sorted by the bare-state photon number as we restrict the first amplitude to be 0.
    indices_branches = np.zeros((n_amps, n_levels), dtype=int)
    indices_branches[0] = np.take_along_axis(np.arange(n_levels), np.argsort(avg_num_array[0]), axis=None)

    # Calculate the indices matrix row by row sequentially. See the example in docstring.
    for i, tracking_idx in enumerate(tracking_indices):
        indices_branches[i+1] = np.take_along_axis(tracking_idx, indices_branches[i], axis=-1)
    
    # Use the indices matrix to sort the Floquet quasienergies and modes.
    quasienergies_branches = np.take_along_axis(quasienergies_array, indices_branches, axis=-1)
    mode_mat_branches = np.take_along_axis(mode_mat_array, indices_branches[:, None], axis=-1)
    avg_num_branches = np.take_along_axis(avg_num_array, indices_branches, axis=-1)

    return quasienergies_branches, mode_mat_branches, avg_num_branches


def fold_out_branch(branch: np.ndarray, f_d: float, threshold: float = None) -> np.ndarray:
    """
    Fold a sorted floquet branch outside the first Brillouin zone.
    It helps avoid the sudden jump from -f_d to +f_d, where f_d is the drive frequency.
    The folded-out branch will typically be used to calculate second derivative, \
        so the whether the other branches are folded doesn't matter.

    Parameters:
    branch: a sorted branch with shape (n_amps, ).
    f_d: drive frequency, the size of the first Brillouin zone.
    threshold: the threshold of the difference in adjacent quasienerigy \
        to determine whether there is a sudden jump or not.
    """
    if threshold is None: threshold = f_d * 0.9
    branch_diff_abs = np.abs(np.diff(branch))

    if branch_diff_abs.max() < threshold: return branch

    idx = np.argmax(branch_diff_abs >= threshold)
    sign = -1 if branch[idx+1] > branch[idx] else 1
    branch[idx+1:] += sign * f_d
    return branch


def identify_target_level(qe: np.ndarray, avc_idx: int, level: int, 
                          second_derivative: float, max_gap: float = 100) -> int:
    """
    Identify the target level at an crossing without knowing if they are avoided.
    The target level should have close value and second derivative of quasienergies \
        at the crossing compared to the current branch.

    Parameters:
    qe: sorted quasienergies with shape (n_amps, n_levels).
    avc_idx: the index of the avoided crossing.
    level: the index of the current branch.
    second_derivative: the second derivative of quasienergies at the crossing of current branch.
    max_gap: the maximum allowed gap between the current branch and the target branch.
    """
    # We first identify levels that are close to (including) the current one.
    close_levels = np.where(
        np.abs(qe[avc_idx] - qe[avc_idx, level]) < max_gap
    )[0].tolist()

    if len(close_levels) > 1:
        # Calculate the second derivatives of all branches at this index.
        second_derivative_all_branches = np.abs(qe[avc_idx+1] - 2*qe[avc_idx] + qe[avc_idx-1])

        # We then identify levels whose 2nd derivatives differ by less than 20%.
        close_d2x_levels = np.where(
            np.abs(second_derivative_all_branches - second_derivative) 
            < (second_derivative / 5)
        )[0].tolist()
        close_d2x_levels.remove(level)

        # We use the intersection of the two sets to identify the target level.
        intersection = set(close_levels).intersection(set(close_d2x_levels))
        assert len(intersection) == 1, 'Algorithm failed to extract the target level.'
        target_level = intersection.pop()
    else:
        target_level = close_levels[0]
    return target_level


def get_gap(qe: np.ndarray, avc_idx: int, level: int, target_level: int) -> float:
    """
    Determin if there is a normal crossing or an avoided crossing at the index.
    Calculate the gap size when there is an avoided crossing.
    If it is not an avoided crossing, return 0.

    We check if they form an avoided crossing by asking whether the qe of one branch is \
        always larger (smaller) than the other branch, both before and after the crossing.
    """
    sign = np.sign(
        (qe[avc_idx-1, target_level] - qe[avc_idx-1, level]) 
        * (qe[avc_idx+1, target_level] - qe[avc_idx+1, level]) 
    )
    return float(np.abs(qe[avc_idx, target_level] - qe[avc_idx, level])) if sign > 0 else 0


def get_avc_indices(quasienergies_branches: np.ndarray, f_d: float, level: int = 0,
                    min_second_derivative: float = 1e-4, max_gap: float = 100,
                    min_avc_indices_distance: int = 200) -> tuple[list[int], list[float]]:
    """
    Find the indices of all avoided crossings along a diabatic branch.

    Parameters:
    quasienergies_branches: sorted quasienergies with shape (n_amps, n_levels).
    f_d: drive frequency, the size of the first Brillouin zone.
    level: the index of the diabatic branch to analyze.
    min_second_derivative: the minimum second derivative to identify a crossing.
    max_gap: the maximum allowed gap between the current branch and the target branch.
    min_avc_indices_distance: the minimum allowed indices distance \
        between two adjacent avoided crossings to avoid double counting.

    The default parameters are for a 4.2GHz transmon driven at 6.3GHz.
    The photons are np.linspace(0, 1500, 15001) with g/2pi ~ 40 MHz.
    """
    qe = quasienergies_branches
    avc_indices = []
    avc_gaps = []

    # Identify the first avoided crossing
    branch = fold_out_branch(qe[:, level], f_d)
    second_derivatives = np.abs(np.diff(np.diff(branch)))
    avc_idx = 0

    while second_derivatives.max() >= min_second_derivative:
        # When there is a peak in second derivatives, there must be an crossing.
        # Without know if they are avoided, we identify the target level first.
        peaks, _ = find_peaks(second_derivatives, height=min_second_derivative)
        peaks = peaks.tolist()
        if len(peaks) == 0: break  # A half peak in the tail may not be identified by scipy.
        avc_idx += peaks[0] + 1
        target_level = identify_target_level(qe, avc_idx, level, second_derivatives[peaks[0]], max_gap)
        
        # If they form an avoided crossing, we append it and switch to new branch.
        if (gap := get_gap(qe, avc_idx, level, target_level)) != 0:
            avc_indices.append(avc_idx)
            avc_gaps.append(gap)
            level = target_level

        # Step forward to avoid double counting.
        avc_idx += min_avc_indices_distance
        if qe.shape[0] - avc_idx < 3: break  # If there are not enough qe left.

        branch = fold_out_branch(qe[avc_idx:, level], f_d)
        second_derivatives = np.abs(np.diff(np.diff(branch)))
    return avc_indices, avc_gaps