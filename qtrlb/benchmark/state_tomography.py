import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from qtrlb.config.config import MetaManager
from qtrlb.calibration.calibration import Scan
from qtrlb.benchmark.state_tomography_tools import TOMOGRAPHY_GATE_SETS, calculate_single_qudit_density_matrix




class StateTomography(Scan):
    """ State tomography for multiple qudit system.
        Require all drive_qubits to use same subspace.
        Any state preparation or special gate before tomography can done by pre_gate.
        Allow passing in a gate_set with minimal key 'd' and 'gates', or a string of built-in gate_set.
    """
    def __init__(
            self,
            cfg: MetaManager,
            drive_qubits: str | list[str],
            readout_resonators: str | list[str],
            subspace: str | list[str],
            gate_set: str | dict,
            main_tones: str | list[str] = None,
            pre_gate: dict[str: list[str]] = None,
            post_gate: dict[str: list[str]] = None,
            n_seqloops: int = 1000):
        
        super().__init__(
            cfg=cfg, 
            drive_qubits=drive_qubits, 
            readout_resonators=readout_resonators, 
            scan_name='StateTomography', 
            x_plot_label='',
            x_plot_unit='arb', 
            x_start=1, 
            x_stop=1, 
            x_points=1,
            subspace=subspace,
            main_tones=main_tones,
            pre_gate=pre_gate,
            post_gate=post_gate,
            n_seqloops=n_seqloops)
        
        assert self.classification_enable, 'STomo: Please turn on classification.'
        assert all(ss == self.subspace[0] for ss in self.subspace), 'STomo: All subspace must be same.'
        self.gate_set = gate_set
        self.generate_tomography_gates()

        # Set attributes for making process_data, set_acquisition and plot_population work.
        self.n_tomography_gates = len(self.tomography_gates_list)
        self.x_points = self.n_tomography_gates
        self.num_bins = self.n_seqloops * self.x_points
        self.x_values = np.arange(self.x_points)


    def generate_tomography_gates(self) -> None:
        """
        Generate all tomography gates which will be added to sequence.
        The specific gates depend on self.gate_set, and len(self.drive_qubits).
        We will return self.tomography_gates_list, which is a list of dictionary.
        Each dictionary is one of the all possible qubit-gate combination.

        Example: two qubit ['Q0', 'Q1'] with gate_set ['I', 'X90_01', 'Y90_01']
        self.tomography_gates_list = [
            {'Q0': ['I'], 'Q1': ['I']},
            {'Q0': ['I'], 'Q1': ['X90_01']},
            {'Q0': ['I'], 'Q1': ['Y90_01']},
            {'Q0': ['X90_01'], 'Q1': ['I']},
            {'Q0': ['X90_01'], 'Q1': ['X90_01']},
            {'Q0': ['X90_01'], 'Q1': ['Y90_01']},
            {'Q0': ['Y90_01'], 'Q1': ['I']},
            {'Q0': ['Y90_01'], 'Q1': ['X90_01']},
            {'Q0': ['Y90_01'], 'Q1': ['Y90_01']},
        ]
        
        Note from Zihao(09/16/2023):
            By calling this function in __init__, we don't need to change run().
            It also helps to pop error earlier when get_set and subspace is not consistent.
        """
        # Get gate set and check dimension.
        if isinstance(self.gate_set, str): self.gate_set = TOMOGRAPHY_GATE_SETS[self.gate_set]
        assert int(self.subspace[0][-1]) + 1 == self.gate_set['d'], \
            f"STomo: This gate set only work on {self.gate_set['d']} levels. Please check subspace."
        
        # Create tomography gates list.
        self.tomography_gates_list = []
        for single_combination in product(self.gate_set['gates'], repeat=len(self.drive_qubits)):
            self.tomography_gates_list.append(
                {q: single_combination[i] for i, q in enumerate(self.drive_qubits)}
            )


        # Reference: Original syntax by Berkeley and Ray.
        # [dict(zip(qubits, p)) for p in product(self.gate_set['gate'], repeat=len(qubits))]


    def make_sequence(self):
        """
        Please refer to Scan.make_sequence for general structure and example.
        Here I have to flatten the x loop for different tomography gates.
        """
        self.sequences = {tone: {} for tone in self.tones}        
        self.set_waveforms_acquisitions()
        self.init_program()
        self.add_seqloop()
        self.add_main()
        self.end_seqloop()


    def add_main(self):
        """
        Add main sequence to sequence program.
        We will "loop" over different tomography gate by flatting them.
        Then seq_loop and py_loop will repeat these circuit at outer layer.
        """
        add_label = False  # Must be False to avoid label confliction.
        concat_df = False  # Nothing should go wrong if True, but it's very slow.        

        for i, tomography_gate in enumerate(self.tomography_gates_list):
            name = f'STomo{i}'
            self.add_sequence_start()
            self.add_relaxation(label=i)
            if self.heralding_enable: self.add_heralding(name+'HRD', add_label=add_label, concat_df=concat_df)
            self.add_gate(self.subspace_gate, 'Subspace', add_label=add_label, concat_df=concat_df)
            self.add_gate(self.pre_gate, 'Pregate', add_label=add_label, concat_df=concat_df)
            self.add_gate(tomography_gate, name, add_label=add_label, concat_df=concat_df)
            self.add_gate(self.post_gate, 'Postgate', add_label=add_label, concat_df=concat_df)
            self.add_readout(name+'RO', add_label=add_label, concat_df=concat_df)
            self.add_sequence_end()


    def process_data(self):
        """
        Multiplexing readout for qudits (A, B, C ...) is represented by measurement operators:
        M_ijk = Mi_A ⊗ Mj_B ⊗ Mk_C,
        where i j k is each possible measurement outcome.
        It means we need to find probability under each possible M_ijk, not individual measurement operator.
        We will further process data and find these probabilities here.

        Note from Zihao(09/17/2023):
        Multiple qudit tomography requires multiplexed single shot qudit readout result \
        BEFORE normalization and correction.
        For example, for two qutrits, we need P_00, P_01, P_02, P_10, P_11, P_12, P_20, P_21, P_22, \
        not P_0_A, P_1_A, P_2_A, P_0_B, P_1_B, P_2_B.
        However, I didn't realize that my twotone_mask and twotone_MLE has problem until today.
        Both of them gives biased single shot and biased normalized result.
        It's the corr_matrix and readout correction process help to keep final population unbiased.
        For now, we can only do single qudit tomography.
        """
        return super().process_data()




class SingleQuditStateTomography(StateTomography):
    """ State tomography for single qudit system.
        It should be compatible with any currect twotone readout problem
    """
    def fit_data(self):
        # Consider only single qudit and first tone has same index as the qudit.
        # populations will have shape (n_readout_levels, n_tomograhy_gates).
        populations = self.measurement[f'R{self.drive_qubits[0][1:]}']['to_fit']
        self.density_matrix = calculate_single_qudit_density_matrix(populations, self.tomography_gates_list)


    def plot_main(self):
        """
        Plot single qudit density matrix.
        Ref: https://stackoverflow.com/questions/53590227/3d-histogram-from-a-matrix-of-z-value
        """
        matrix = np.abs(self.density_matrix)
        d = matrix.shape[0]

        fig = plt.figure(figsize=(d, d), dpi=400)
        ax = fig.add_subplot(111, projection='3d')

        x = [i for i in range(d) for _ in range(d)]  # The last 'for' generate inner layer. 
        y = [i for _ in range(d) for i in range(d)]
        z = np.zeros((d**2))
        dx = dy = 0.5 * np.ones((d**2))
        dz = matrix.flatten()
        ax.bar3d(x, y, z, dx, dy, dz)
        ax.set(title=f'{self.datetime_stamp}, Magnitude of Density Matrix', xlabel='row', ylabel='column')
        ax.title.set_size(8)
        fig.savefig(os.path.join(self.data_path, 'Density_Matrix.png'))

        r = f'R{self.drive_qubits[0][1:]}'
        self.figures = {r: fig}

