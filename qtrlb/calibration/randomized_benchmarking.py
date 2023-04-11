import os
import json
import numpy as np
from lmfit import Model
from copy import deepcopy
from qtrlb.calibration.calibration import Scan
from qtrlb.processing.fitting import ExpModel2
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
                 n_random: int = 30,
                 subspace: str = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int = 0,
                 fitmodel: Model = ExpModel2):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits, 
                         readout_resonators=readout_resonators, 
                         scan_name='RB1QB', 
                         x_plot_label='Number of Clifford Gates',
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
        
        
    def run(self, 
            experiment_suffix: str = '',
            n_pyloops: int = 1):
        """
        Because the 16384 instruction limit of the instrument, we cannot run all random at once.
        Instead, we will use measurement/measurements trick to run different random sequence.
        The fit and plot will happen after we get all measurements.
        The fitting result and figure file will be saved in the last measurement folder.
        """
        self.n_pyloops = n_pyloops
        self.n_reps = self.n_seqloops * self.n_pyloops
        self.attrs = deepcopy(self.__dict__)
        
        for i in range(self.n_random):
            self.experiment_suffix = experiment_suffix + f'_Random_{self.n_runs}'
            self.make_sequence() 
            self.save_sequence()
            self.cfg.DAC.implement_parameters(self.drive_qubits, self.readout_resonators, self.jsons_path) 
            self.make_exp_dir()  # It also save a copy of yamls and jsons there.
            self.acquire_data()  # This is really run the thing and return to the IQ data in self.measurement.
            self.cfg.data.save_measurement(self.data_path, self.measurement, self.attrs)
            self.process_data()
            self.cfg.data.save_measurement(self.data_path, self.measurement, self.attrs)
            self.n_runs += 1
            self.measurements.append(self.measurement)
            self.plot_IQ()
            if self.classification_enable: self.plot_populations()
            
        self.fit_data()
        self.plot()

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
                                                                          n_random=1))
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


    def save_sequence(self, jsons_path: str = None):
        """
        Also save each randomized sequence to their experiment folder.
        """
        super().save_sequence(jsons_path=jsons_path)
        
        if jsons_path is None: return
        file_path = os.path.join(jsons_path, 'RB_sequence.json')
        
        with open(file_path, 'w', encoding='utf-8') as file:
            both_sequences = {'Clifford_sequences': self.Clifford_sequences,
                              'primitive_sequences': self.primitive_sequences}
            json.dump(both_sequences, file, indent=4)


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
        

        for j in range(self.x_points):
            primitive_sequence = self.primitive_sequences[j][0]  # For only single random.
            
            pulse = {q: primitive_sequence for q in self.drive_qubits}
            name = f'RBpoint{j}'
            lengths = [qubit_pulse_length if not gate.startswith('Z') or gate.startswith('I') else 0
                       for gate in primitive_sequence]
            
            self.add_sequence_start()
            self.add_wait(name+'RLX', relaxation_length, add_label=add_label, concat_df=concat_df)
            if heralding: self.add_heralding(name+'HRD', add_label=add_label, concat_df=concat_df)
            self.add_pulse(self.subspace_pulse, 'Subspace', add_label=add_label, concat_df=concat_df)
            self.add_pulse(pulse, name, lengths, add_label=add_label, concat_df=concat_df)
            self.add_readout(name+'RO', add_label=add_label, concat_df=concat_df)
            self.add_sequence_end()


    def fit_data(self):
        """
        Combine result of all measurements and fit it.
        
        Note from Zihao (04/03/2023):
        Although we should only have 1 resonator, but I still leave the loop here for extendibility. 
        I overwrite the last measurement[r]['to_fit'] to reuse Scan.fit_data. 
        I believe this Scan is special and specific enough to treat it differetly without \
        considering too much of generality.
        """
        self.data_all_randoms = {}

        for r in self.readout_resonators:
            self.data_all_randoms[r] = []
            
            for measurement in self.measurements:
                self.data_all_randoms[r].append(measurement[r]['to_fit'])
                
            # It should have shape (n_random, n_levels, x_points).
            self.data_all_randoms[r] = np.array(self.data_all_randoms[r])
            self.measurement[r]['to_fit'] = np.mean(self.data_all_randoms[r], axis=0)
        
        super().fit_data()


    def plot(self):
        self.plot_main(text_loc='lower left')
        
        # Plot result of each random sequence.
        for j, r in enumerate(self.readout_resonators):
            ax = self.figures[r].get_axes()[0]
            
            for i in range(self.n_random):
                ax.plot(self.x_values / self.x_unit_value, 
                        self.data_all_randoms[r][i][self.level_to_fit[j]], 
                        'b.')

            self.figures[r].canvas.draw()











