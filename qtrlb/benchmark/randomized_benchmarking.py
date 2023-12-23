import os
import json
import numpy as np
from lmfit import Model
from abc import ABCMeta, abstractmethod
from qtrlb.config.config import MetaManager
from qtrlb.calibration.calibration import Scan
from qtrlb.calibration.scan_classes import Spectroscopy
from qtrlb.processing.fitting import ExpModel2
from qtrlb.benchmark.RB1QB_tools import generate_RB_Clifford_gates, generate_RB_primitive_gates




class RB1QBBase(Scan, metaclass=ABCMeta):
    """ Base class for Randomized Benchmarking type of experiments.
        User must overload the __init__, make_sequence and add_main for child class.
        A integer n_random need to be specified during __init__
    """
    @abstractmethod
    def __init__(self, n_random: int):
        return
        
        
    def run(self, 
            experiment_suffix: str = '',
            n_pyloops: int = 1,
            process_kwargs: dict = None,
            fitting_kwargs: dict = None,
            plot_kwargs: dict = None):
        """
        Because the 16384 instruction limit of the instrument, we cannot run all random at once.
        Instead, we will use measurement/measurements trick to run different random sequences.
        The fit and plot will happen after we get all measurements.
        It means we shouldn't call self.run() twice, instead, we shuold redo instantiation.
        This is similar to ReadoutLengthAmpScan.
        """
        # Check attributes
        assert hasattr(self, 'n_random'), 'RB1QBBase: Please specify n_random.'
        self.measurements = []  # Allow user to call run() without redo instantiation.

        # Set attributes as usual
        self.set_running_attributes(experiment_suffix, n_pyloops, process_kwargs, fitting_kwargs, plot_kwargs)

        # Make the main folder, but not save sequence here since we don't have it yet.
        self.cfg.data.make_exp_dir(experiment_type='_'.join([*self.main_tones_, self.scan_name]),
                                   experiment_suffix=self.experiment_suffix)
        self.main_data_path = self.cfg.data.data_path
        self.datetime_stamp = self.cfg.data.datetime_stamp
        self.cfg.save(yamls_path=self.cfg.data.yamls_path, verbose=False)

        # Loop over each random
        for i in range(self.n_random):
            # Make the sub folder, self.data_path will be updated here.
            self.data_path = os.path.join(self.main_data_path, f'Random_{i}')
            os.makedirs(os.path.join(self.data_path, 'Jsons'))
            for rt_ in self.readout_tones_: os.makedirs(os.path.join(self.data_path, 'IQplots', rt_))
            
            # Run as usual, but using the new self.data_path.
            self.make_sequence() 
            self.save_sequence()
            self.save_sequence(jsons_path=os.path.join(self.data_path, 'Jsons'))
            self.upload_sequence()
            self.acquire_data()
            self.save_data()
            self.process_data()
            self.save_data()
            self.measurements.append(self.measurement)
            if self.cfg['variables.common/plot_IQ']: self.plot_IQ()
            if self.classification_enable: self.plot_populations()
            print(f'Random {i} finished!')

        # Fit, save and plot full result after all measurements.
        self.data_path = self.main_data_path
        self.fit_data()
        self.save_data()
        self.plot_full_result()
        self.n_runs += 1


    @abstractmethod
    def make_sequence(self):
        return


    @abstractmethod
    def add_main(self):
        return


    def save_sequence(self, jsons_path: str = None):
        """
        Save Q1ASM sequences and also each randomized gates sequence to their sub folders.
        """
        super().save_sequence(jsons_path=jsons_path)
        
        if ( not hasattr(self, 'Clifford_gates') ) or ( jsons_path is None ): return
        both_sequences = {'Clifford_gates': self.Clifford_gates,
                          'primitive_gates': self.primitive_gates}
        
        with open(os.path.join(jsons_path, 'RB_sequence.json'), 'w', encoding='utf-8') as file:
            json.dump(both_sequences, file, indent=4)


    def fit_data(self):
        """
        Combine result of all randoms and fit it.
        
        Note from Zihao (04/03/2023):
        I overwrite the LAST measurement[r] to reuse Scan.fit_data and save it to main_data_path.
        However, the last measurement.hdf5 still keep last random data correctly.
        It's because there is no self.save_data() after self.fit_data() 
        I believe this Scan is special and specific enough to treat it differetly without \
        considering too much of generality.
        """
        for rr in self.readout_resonators:
            # It will have shape (n_random, n_levels, x_points).
            data_all_random = np.array([measurement[rr]['to_fit'] for measurement in self.measurements])

            self.measurement[rr] = {
                'all_random': data_all_random,
                'to_fit': np.mean(data_all_random, axis=0)
            }
        super().fit_data()


    def plot_full_result(self):
        """
        Plot each random result with their average and the fitting for average.
        """
        self.plot_main(text_loc='lower left')
        
        # Plot result of each random sequence.
        for j, rr in enumerate(self.readout_resonators):
            ax = self.figures[rr].get_axes()[0]  # figure.get_axes always return list of axis object.
            
            for i in range(self.n_random):
                ax.plot(self.x_values / self.x_unit_value, 
                        self.measurement[rr]['all_random'][i][self.level_to_fit[j]], 
                        'r.', alpha=0.1)

            self.figures[rr].savefig(os.path.join(self.main_data_path, f'{rr}.png'))
            self.figures[rr].canvas.draw()


class RB1QB(RB1QBBase):
    """ Randomized Benchmarking (RB) for single qubit or any single two-level subspace.

        Note from Zihao(09/09/2023):
        This is not design for simultaneous RB on multiple single qubit, \
        which is better metric that include crosstalk, stark shift, idling decoherence.
        It's not about the two-qubit gate, it's because here I and Z gate don't take real time.
        It will break synchronization when we run it on multiple qubits simultaneously.
        The pulse.py accept finite time of I and Z.
        So if we want it, we need to change optimize_circuit in RB1QB_tools.
        And also the way we generate self.Clifford_gates since it's different between qubits.
    """
    def __init__(
            self,
            cfg: MetaManager,
            drive_qubits: str | list[str],
            readout_tones: str | list[str],
            n_gates_start: int,
            n_gates_stop: int,
            n_gates_points: int,
            n_random: int = 30,
            subspace: str | list[str] = None,
            main_tones: str | list[str] = None,
            pre_gate: dict[str: list[str]] = None,
            post_gate: dict[str: list[str]] = None,
            n_seqloops: int = 1000,
            level_to_fit: int | list[int] = None,
            fitmodel: Model = ExpModel2):
        
        super(RB1QBBase, self).__init__(
            cfg=cfg, 
            drive_qubits=drive_qubits, 
            readout_tones=readout_tones, 
            scan_name='RB1QB', 
            x_plot_label='Number of Clifford Gates',
            x_plot_unit='arb', 
            x_start=n_gates_start, 
            x_stop=n_gates_stop, 
            x_points=n_gates_points,
            subspace=subspace,
            main_tones=main_tones,
            pre_gate=pre_gate,
            post_gate=post_gate,
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
        Notice the sequence here is for single random.
        """
        self.sequences = {tone: {} for tone in self.tones}         
        self.set_waveforms_acquisitions()
        
        self.Clifford_gates = []
        self.primitive_gates = []
        for i, n_gates in enumerate(self.x_values):
            self.Clifford_gates.append(generate_RB_Clifford_gates(n_gates))
            self.primitive_gates.append(generate_RB_primitive_gates(self.Clifford_gates[i]))            
        # =============================================================================
        # Here self.Clifford_gates is nested list with shape (self.x_points, n_gates+1)
        # Each element is a string of Clifford gate name, such as 'S', '-C_-xyz'. 
        # The primitive_gates replace the Clifford gate string with primitive gate string.
        # Its shape is uncertain.
        # =============================================================================
        
        self.init_program()
        self.add_seqloop()
        self.add_main()
        self.end_seqloop()


    def add_main(self):
        """
        Add main sequence to sequence program.
        We will "loop" over different gate length by flatting them.
        Then seq_loop and py_loop will repeat these circuit.
        Randomization will be the outermost layer.
        """
        add_label = False  # Must be False to avoid label confliction.
        concat_df = False  # Nothing should go wrong if True, but it's very slow.        

        for j in range(self.x_points):
            primitive_gate = self.primitive_gates[j]

            main_gate = {}
            for tone in self.main_tones:
                qubit, subspace = tone.split('/')
                main_gate[qubit] = [f'{gate}_{subspace}' for gate in primitive_gate]
            
            name = f'RBpoint{j}'
            lengths = [self.qubit_pulse_length_ns 
                       if not gate.startswith('Z') or gate.startswith('I') else 0
                       for gate in primitive_gate]
            
            self.add_sequence_start()
            self.add_relaxation(label=j)
            if self.heralding_enable: self.add_heralding(name+'HRD', add_label=add_label, concat_df=concat_df)
            self.add_gate(self.subspace_gate, 'Subspace', add_label=add_label, concat_df=concat_df)
            self.add_gate(self.pre_gate, 'Pregate', add_label=add_label, concat_df=concat_df)
            self.add_gate(main_gate, name, lengths, add_label=add_label, concat_df=concat_df)
            self.add_gate(self.post_gate, 'Postgate', add_label=add_label, concat_df=concat_df)
            self.add_readout(name+'RO', add_label=add_label, concat_df=concat_df)
            self.add_sequence_end()


class RB1QBDetuningSweep(RB1QBBase, Spectroscopy):
    """ Pulse detuning sweep based on randomized gate sequence.
        We will set fixed number of gates and randoms and variable frequency.
        We can use this class to find the optimal frequency for getting high RB fidelity.
    """
    def __init__(
            self,
            cfg: MetaManager,
            drive_qubits: str | list[str],
            readout_tones: str | list[str],
            detuning_start: float, 
            detuning_stop: float, 
            detuning_points: int, 
            n_gates: int = 100,
            n_random: int = 30,
            subspace: str | list[str] = None,
            main_tones: str | list[str] = None,
            pre_gate: dict[str: list[str]] = None,
            post_gate: dict[str: list[str]] = None,
            n_seqloops: int = 1000,
            level_to_fit: int | list[int] = None,
            fitmodel: Model = None):
        
        super(Spectroscopy, self).__init__(
            cfg=cfg, 
            drive_qubits=drive_qubits, 
            readout_tones=readout_tones, 
            scan_name='RB1QBDetuningSweep', 
            x_plot_label='Pulse Detuning',
            x_plot_unit='MHz', 
            x_start=detuning_start, 
            x_stop=detuning_stop, 
            x_points=detuning_points,
            subspace=subspace,
            main_tones=main_tones,
            pre_gate=pre_gate,
            post_gate=post_gate,
            n_seqloops=n_seqloops,
            level_to_fit=level_to_fit,
            fitmodel=fitmodel)
        
        self.n_gates = n_gates
        self.n_random = n_random


    def make_sequence(self):
        """
        Generate self.sequence and self.primitive_gates
        Here self.Clifford_gates is a list of string of Clifford gate name, generate_RB_primitive_gates
        The primitive_gates replace the Clifford gate string with primitive gate string.
        Its shape is uncertain because of the optimize_circuit.
        """
        self.Clifford_gates = generate_RB_Clifford_gates(self.n_gates)
        self.primitive_gates = generate_RB_primitive_gates(self.Clifford_gates)    

        super(RB1QBBase, self).make_sequence()


    def add_main(self):
        """
        Add main randomized gates and step of x_values.
        We then replace the frequency of main_tones to the register R4 (specific x_values).
        """
        main_gate = {}
        for tone in self.main_tones:
            qubit, subspace = tone.split('/')
            main_gate[qubit] = [f'{gate}_{subspace}' for gate in self.primitive_gates]

        lengths = [self.qubit_pulse_length_ns if not gate.startswith('Z') or gate.startswith('I') else 0
                  for gate in self.primitive_gates]
        
        self.add_gate(main_gate, 'RB', lengths, add_label=False, concat_df=False)

        for tone in self.main_tones:
            self.sequences[tone]['program'] += f"""
                    add              R4,{self.frequency_translator(self.x_step)},R4
            """

            tone_dict = self.cfg[f'variables.{tone}']
            freq = round((tone_dict['mod_freq'] + tone_dict['pulse_detuning']) * 4)

            old_str = f'set_freq         {freq}'
            new_str = f'set_freq         R4'
            self.sequences[tone]['program'] = self.sequences[tone]['program'].replace(old_str, new_str)

    
    def fit_data(self):
        return super(Spectroscopy, self).fit_data()



class RB1QBAmplitudeSweep(RB1QBBase, Spectroscopy):
    """ Pulse amplitude sweep based on randomized gate sequence.
        We will set fixed number of gates and randoms and variable frequency.
        We can use this class to find the optimal frequency for getting high RB fidelity.
    """
    def __init__(
            self,
            cfg: MetaManager,
            drive_qubits: str | list[str],
            readout_tones: str | list[str],
            amp_start: float, 
            amp_stop: float, 
            amp_points: int, 
            n_gates: int = 100,
            n_random: int = 30,
            subspace: str | list[str] = None,
            main_tones: str | list[str] = None,
            pre_gate: dict[str: list[str]] = None,
            post_gate: dict[str: list[str]] = None,
            n_seqloops: int = 1000,
            level_to_fit: int | list[int] = None,
            fitmodel: Model = None):
        
        super(RB1QBBase, self).__init__(
            cfg=cfg, 
            drive_qubits=drive_qubits, 
            readout_tones=readout_tones, 
            scan_name='RB1QBAmplitudeSweep', 
            x_plot_label='Amplitude',
            x_plot_unit='arb', 
            x_start=amp_start, 
            x_stop=amp_stop, 
            x_points=amp_points,
            subspace=subspace,
            main_tones=main_tones,
            pre_gate=pre_gate,
            post_gate=post_gate,
            n_seqloops=n_seqloops,
            level_to_fit=level_to_fit,
            fitmodel=fitmodel)
        
        self.n_gates = n_gates
        self.n_random = n_random