# I try my best to follow the code style of scqubits, so that it can be easily
# integrated into the scqubits library in the future --Zihao
############################################################################

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy as sp
import qutip as qt
import scqubits as scq

from numpy import ndarray

import scqubits.core.qubit_base as base
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization

# - Cooper pair box / transmon with addtional Josephson Harmonics -


class HarmonicTransmon(scq.Transmon):
    r"""Class for the Cooper-pair-box and transmon qubit with additional Harmonics terms. 
    The Hamiltonian is represented in dense form in the number basis,
    :math:`H_\text{CPB}=4E_\text{C}(\hat{n}-n_g)^2-\frac{E_\text{J}}{2}(
    |n\rangle\langle n+1|+\text{h.c.})`.
    Initialize with, for example::

        Transmon(EJ=[1.0. -0.01], EC=2.0, ng=0.2, ncut=30)

    Parameters
    ----------
    EJ:
        Josephson energies
    EC:
        charging energy
    ng:
        offset charge
    ncut:
        charge basis cutoff, `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    EJ = descriptors.WatchedProperty(list, "QUANTUMSYSTEM_UPDATE")
    n_harmonics = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: list | np.ndarray,
        EC: float,
        ng: float,
        ncut: int,
        n_harmonics: int = 1,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
        evals_method: Union[Callable, str, None] = None,
        evals_method_options: Union[dict, None] = None,
        esys_method: Union[Callable, str, None] = None,
        esys_method_options: Union[dict, None] = None,
    ) -> None:
        base.QubitBaseClass.__init__(
            self,
            id_str=id_str,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.n_harmonics = n_harmonics
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-np.pi, np.pi, 151)
        self._default_n_range = (-5, 6)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {"EJ": [15.0], "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}
    
    def _hamiltonian_diagonal(self) -> ndarray:
        raise NotImplementedError('This method should not be used as in scqubits.Transmon')

    def _hamiltonian_offdiagonal(self) -> ndarray:
        raise NotImplementedError('This method should not be used as in scqubits.Transmon')
    
    def _evals_calc(self, evals_count: int) -> ndarray:
        evals = np.linalg.eigvalsh(self.hamiltonian())
        return evals[:evals_count]

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        evals, evecs = np.linalg.eigh(self.hamiltonian())
        return evals[:evals_count], evecs[:, :evals_count]
    
    def exp_i_m_phi_operator(
        self, m: int, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """Returns operator :math:`e^{im\\varphi}` in the charge basis"""
        dimension = self.hilbertdim()
        entries = np.repeat(1.0, dimension - m)
        exp_op = np.diag(entries, -1 * m)
        return self.process_op(native_op=exp_op, energy_esys=energy_esys)
    
    def cos_m_phi_operator(
        self, m: int, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """Returns operator :math:`\\cos(m\\varphi)` in the charge basis"""
        cos_op = 0.5 * self.exp_i_m_phi_operator(m)
        cos_op += cos_op.T
        return self.process_op(native_op=cos_op, energy_esys=energy_esys)

    def sin_m_phi_operator(
        self, m: int, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """Returns operator :math:`\\sin(m\\varphi)` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_m_phi_operator(m)
        sin_op += sin_op.conjugate().T
        return self.process_op(native_op=sin_op, energy_esys=energy_esys)

    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """Returns Hamiltonian in charge basis"""
        hamiltonian_mat = 4 * self.EC * np.diag(
            (np.linspace(-1*self.ncut, self.ncut, self.hilbertdim()) - self.ng) ** 2
        )
        for m in range(1, self.n_harmonics+1):
            hamiltonian_mat += -self.EJ[m-1] * self.cos_m_phi_operator(m)
        return self.process_hamiltonian(
            native_hamiltonian=hamiltonian_mat, energy_esys=energy_esys
        )
    
    def potential(self, phi: Union[float, ndarray]) -> ndarray:
        """Transmon phase-basis potential evaluated at `phi`.

        Parameters
        ----------
        phi:
            phase variable value
        """
        potential = 0
        for m in range(1, self.harmonics+1):
            potential += -self.EJ[m-1] * np.cos(m * phi)
        return potential


# - A coupled Harmonic Transmon and resonator -


class CoupledHarmonicTransmonResonator:
    def __init__(self,
                 EJ: list | np.ndarray,
                 EC: float,
                 ng: float,
                 ncut: int,
                 n_harmonics: int,
                 fr_b: float,
                 g: float,
                 r_levels: int):
        self.hmon = HarmonicTransmon(EJ, EC, ng, ncut, n_harmonics)
        self.fr_b = fr_b
        self.g = g
        self.r_levels = r_levels

        
    def hamiltonian(self) -> qt.Qobj:
        """
        Return Hamiltonian of the coupled system.
        """
        H_t = qt.tensor(qt.Qobj(self.hmon.hamiltonian()), qt.identity(self.r_levels))
        H_r = qt.tensor(qt.identity(self.hmon.hilbertdim()), 
                        qt.Qobj(np.diag([0.5 + r * self.fr_b for r in range(self.r_levels)])))
        V = self.g * qt.tensor(qt.Qobj(self.hmon.n_operator()), 
                               1j * (qt.create(self.r_levels) - qt.destroy(self.r_levels)))
        return H_t + H_r + V


    def diagonalize(self):
        """
        Diagonalize the Hamiltonian and get eigenenergies and evecs.
        """
        self.evals, self.evecs = self.hamiltonian().eigenstates()
        self.evals -= self.evals[0]


    def sort_eigenenergies(self, q_photons: int = 3, r_photons: int = 2, max_photons: int = 100):
        _, evecs_hmon = self.hmon.eigensys(evals_count=q_photons+1)

        self.evals_sorted = {}
        for t in range(q_photons+1):
            for r in range(r_photons+1):
                bare_state = qt.tensor(qt.Qobj(evecs_hmon[:, t]), qt.basis(self.r_levels, r))

                max_overlap = 0
                for eval, evec in zip(self.evals[:max_photons], self.evecs[:max_photons]):
                    overlap = abs(evec.dag() * bare_state)
                    if overlap < max_overlap: continue
                    max_overlap, energy = overlap, eval

                self.evals_sorted[(t, r)] = energy


# - inverse eigenvalue problem(IEP) -


def fit_coupled_harmonic_transmon_resonator(
        fq_meas: list | ndarray, 
        fr_meas: list | ndarray,
        n_harmonics: int, 
        n_transitions: int, 
        fr_eax: int = 100,
        ng: float = 0.5,
        r_levels: int = 8,
        max_photons: int = 40,
        **minize_kwargs
    ) -> sp.optimize.OptimizeResult:

    def obj_fun(params):
        chtr = CoupledHarmonicTransmonResonator(
            EJ=params[:n_harmonics], EC=params[n_harmonics], ng=ng, ncut=2*(n_transitions+1)+6, 
            n_harmonics=n_harmonics, fr_b=params[n_harmonics+1], g=params[n_harmonics+2], 
            r_levels=r_levels
        )
        chtr.diagonalize()
        chtr.sort_eigenenergies(q_photons=n_transitions+1, r_photons=2, max_photons=max_photons)

        loss = 0
        for t in range(n_transitions):
            loss += (chtr.evals_sorted[(t+1, 0)] - chtr.evals_sorted[(t, 0)] 
                    - fq_meas[t])**2
        
        for t in range(2):
            loss += (chtr.evals_sorted[(t, 1)] - chtr.evals_sorted[(t, 0)]
                    - fr_meas[t])**2 * fr_eax
        return loss
    
    result = sp.optimize.minimize(obj_fun, **minize_kwargs)
    return result