import numpy as np
from lmfit import Model
from qtrlb.calibration.calibration import Scan
from qtrlb.utils.RB1QB_tools import Clifford_gates, Clifford_to_primitive, \
    generate_RB_Clifford_sequences, generate_RB_primitive_sequences
    
    
    
    
class RB1QB(Scan):
    def __init__(self,
                 cfg,
                 drive_qubits: str,
                 readout_resonators: str,
                 n_gates_start: int,
                 n_gates_stop: int,
                 n_gates_points: int,
                 n_random: int = 1,
                 subspace: str = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int = 0,
                 fitmodel: Model = None):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits, 
                         readout_resonators=readout_resonators, 
                         scan_name='RB1QB', 
                         x_plot_label='Number of Clifford Gate',
                         x_plot_unit='arb', 
                         x_start=n_gates_start, 
                         x_stop=n_gates_stop, 
                         x_points=n_gates_points,
                         subspace=subspace,
                         prepulse=None,
                         postpulse=None,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        assert self.x_step.is_integer(), 'All n_gates should be integer.'
        self.x_values = self.x_values.astype(int)
        self.n_random = n_random
        

    def make_sequence(self):
        """
        Please refer to Scan.make_sequence for general structure and example.
        Here I have to flatten the x loop for randomized benchmarking.
        """
        self.sequences = {qudit:{} for qudit in self.qudits}        
        self.set_waveforms_acquisitions()
        
        self.Clifford_sequences = []
        self.primitive_sequences = []
        for i, n_gates in enumerate(self.x_values):
            self.Clifford_sequences.append(generate_RB_Clifford_sequences(Clifford_gates, 
                                                                          n_gates=n_gates, 
                                                                          n_random=self.n_random))
            self.primitive_sequences.append(generate_RB_primitive_sequences(self.Clifford_sequences[i], 
                                                                            Clifford_to_primitive))            
        # =============================================================================
        # Here the Clifford_sequences is list of ndarray. Its length is self.x_points.
        # Inside the list, each ndarray has string('<U7') entries and shape (n_random, n_gates+1), 
        # where n_gates is the corresponding x_value.
        # The primitive_sequences replace the Clifford gate string with primitive gate string.
        # Its shape is uncertain.
        # =============================================================================
        
        self.init_program()
        self.add_seqloop()
        self.add_mainpulse()
        self.end_seqloop()


    def add_mainpulse(self):
        """
        We will loop over all number of clifford first(inner), then loop over all randomization(outer).
        Since randomization is a type of averaging to some extent.
        """
        qubit_pulse_length = round(self.cfg['variables.common/qubit_pulse_length'] * 1e9)
        relaxation_length = round(self.cfg['variables.common/relaxation_time'] * 1e9)
        add_label = False  # Nothing should go wrong if make it True. Just for simplicity.
        concat_df = False  # Nothing should go wrong if make it True. But it's very slow.
        heralding = self.cfg.variables['common/heralding']
        
        for i in range(self.n_random):
            for j in range(self.x_points):
                primitive_sequence = self.primitive_sequences[j][i]
                
                pulse = {q: primitive_sequence for q in self.drive_qubits}
                name = f'RB{i}{j}'
                lengths = [qubit_pulse_length if not gate.startswith('Z') or gate.startswith('I') else 0
                           for gate in primitive_sequence]
                
                self.add_sequence_start()
                self.add_wait(name+'RLX', relaxation_length, add_label, concat_df)
                if heralding: self.add_heralding(name+'HRD', add_label, concat_df)                
                self.add_pulse(pulse, name, lengths, add_label, concat_df)
                self.add_readout(name+'RO', add_label, concat_df)
                self.add_sequence_end()
                print(f'Random Sequence {i} has been generated!')
                
            


    def fit_data(self):
        pass
        
        
    def plot(self):
        pass














