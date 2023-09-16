from itertools import product

import numpy as np
from qtrlb.config.config import MetaManager
from qtrlb.calibration.calibration import Scan
from qtrlb.benchmark.state_tomography_tools import TOMOGRAPHY_GATE_SETS




class StateTomography(Scan):
    """ State tomography for multiple qudit system.
        Require all drive_qubits to use same subspace.
        Any state preparation or special gate before tomography can done by pre_gate.
    """
    def __init__(
            self,
            cfg: MetaManager,
            drive_qubits: str | list[str],
            readout_resonators: str | list[str],
            subspace: str | list[str],
            gate_set_name: str,
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
        
        self.gate_set_name = gate_set_name
        assert all(ss == self.subspace[0] for ss in self.subspace), 'STomo: All subspace must be same.'
        self.generate_tomography_gates()


    def generate_tomography_gates(self) -> None:
        """
        Generate all tomography gates which will be added to sequence.
        The specific gates depend on self.gate_set_name, and len(self.drive_qubits).
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
        gate_set_dict = TOMOGRAPHY_GATE_SETS[self.gate_set_name]
        assert int(self.subspace[0][-1]) + 1 == gate_set_dict['d'], \
            f"STomo: gate set {self.gate_set_name} only work on {gate_set_dict['d']} levels. Please check subspace."
        
        # Create tomography gates list.
        self.tomography_gates_list = []
        for single_combination in product(gate_set_dict['gate'], repeat=len(self.drive_qubits)):
            self.tomography_gates_list.append(
                {q: single_combination[i] for i, q in enumerate(self.drive_qubits)}
            )
        self.n_tomography_gates = len(self.tomography_gates_list)

        # Reference: Original syntax by Berkeley and Ray.
        # [dict(zip(qubits, p)) for p in product(gate_set_dict['gate'], repeat=len(qubits))]


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


    def fit_data(self):
        return
