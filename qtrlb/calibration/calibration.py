import os
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qtrlb.utils.units as u
from copy import deepcopy
from matplotlib.offsetbox import AnchoredText
from lmfit import Model
from qtrlb.config.config import MetaManager
from qtrlb.utils.tone_utils import tone_to_qudit, find_subtones, split_subspace
from qtrlb.utils.waveforms import get_waveform
from qtrlb.utils.pulses import dict_to_DataFrame, gate_transpiler, pulse_interpreter
from qtrlb.utils.general_utils import make_it_list
from qtrlb.processing.fitting import fit
from qtrlb.processing.plotting import COLOR_LIST, plot_IQ




class Scan:
    """ Base class for all parameter-sweep experiment.
        The framework of how experiment flow will be constructed here.
        It should be used as parent class of specific scan rather than being instantiated directly.
        In __init__(), we shape the input parameter to better structure.
        We will check those attribute to ensure they have correct shape/values.
        
        Attributes:
            cfg: A MetaManager.
            drive_qubits: 'Q2', or ['Q3', 'Q4'].
            readout_tones: 'R3/a' or ['R1/a', 'R1/b', 'R5/a'].
            scan_name: 'drive_amplitude', 't1', 'ramsey'.
            x_plot_label: The label of x axis on main plot.
            x_plot_unit: The unit of x axis on main plot.
            x_start: Initial value of parameter-sweep, always in Hz/Second etc without any prefix.
            x_stop: Final value of parameter-sweep, always in Hz/Second etc without any prefix.
            x_points: Number of points of parameter_sweep. Start and stop points will be both included.
            subspace: '12' or ['01', '12']. The highest subspace that experiment will reach to.
                    It will decide the sequencer involved. Length must be same as drive_qubits.
                    Each element should be one-to-one associated to each qubit in order.
                    If main_tones is None, then it will also be the subspace that experiment will work on.
            main_tones: 'Q0/12' or ['Q0/01', 'Q0/12']. The tones that experiment will work on if specified.      
            pre_gate: {'Q0': ['X180_01'], 'Q1': ['X90_12', 'Y90_12']}. They apply just before main sequence.
            post_gate: {'Q0': ['X180_01'], 'Q1': ['X90_12', 'Y90_12']}. They apply just after main sequence.
            n_seqloops: Number of repetition inside sequence program. Total repetition will be self.n_reps.
            level_to_fit: 0, 1 or [0,1,0,0]. Length should be same as readout_resonators.
            fitmodel: A Model from lmfit. Although it should be better to pick from qtrlb.processing.fitting.
    """
    color_list = COLOR_LIST


    def __init__(self, 
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 scan_name: str,
                 x_plot_label: str, 
                 x_plot_unit: str, 
                 x_start: float, 
                 x_stop: float, 
                 x_points: int, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None,
                 **attr_kwargs):

        self.cfg = cfg
        self.drive_qubits = make_it_list(drive_qubits)
        self.readout_tones = make_it_list(readout_tones)
        self.scan_name = scan_name
        self.x_plot_label = x_plot_label
        self.x_plot_unit = x_plot_unit
        self.x_start = x_start
        self.x_stop = x_stop
        self.x_points = x_points
        self.subspace = make_it_list(subspace, ['01' for _ in self.drive_qubits])
        self.main_tones = make_it_list(main_tones)
        self.pre_gate = pre_gate if pre_gate is not None else {}
        self.post_gate = post_gate if post_gate is not None else {}
        self.n_seqloops = n_seqloops
        self.level_to_fit = make_it_list(level_to_fit, 
                                         [self.cfg[f'variables.{rr}/lowest_readout_levels'] 
                                          for rr in self.readout_resonators])
        self.fitmodel = fitmodel
        
        self.n_runs = 0
        self.measurements = []
        self.classification_enable = self.cfg.variables['common/classification']
        self.heralding_enable = self.cfg.variables['common/heralding']
        self.customized_data_process = self.cfg.variables['common/customized_data_process']
        self.x_values = np.linspace(self.x_start, self.x_stop, self.x_points)
        self.x_step = (self.x_stop - self.x_start) / (self.x_points-1) if self.x_points != 1 else 0 
        self.num_bins = self.n_seqloops * self.x_points
        self.jsons_path = os.path.join(self.cfg.working_dir, 'Jsons')
        self.qubit_pulse_length_ns = round(self.cfg['variables.common/qubit_pulse_length'] * 1e9)
        self.resonator_pulse_length_ns = round(self.cfg['variables.common/resonator_pulse_length'] * 1e9)
        # Last two lines just for convenience.
        
        for key, value in attr_kwargs.items(): setattr(self, key, value)
        self.check_attribute()
        self.make_tones_list()
        
        self.x_unit_value = getattr(u, self.x_plot_unit)
        self.subspace_gate = {tone.split('/')[0]: [f'X180_{l}{l+1}' for l in range(split_subspace(tone.split('/')[1])[0])] 
                              for tone in self.main_tones if tone.startswith('Q')}
        self.readout_gate = {rr: ['RO_' + '_'.join(tone.split('/')[1] for tone in find_subtones(rr, self.readout_tones))] 
                             for rr in self.readout_resonators}

        
    def run(self, 
            experiment_suffix: str = '',
            n_pyloops: int = 1,
            process_kwargs: dict = None,
            fitting_kwargs: dict = None,
            plot_kwargs: dict = None):
        """
        Make sequence, implement parameter to qblox instrument.
        Then Run the experiment and acquire data, process data, fit and plot.
        User can call it multiple times without instantiate the Scan class again.
        
        Attributes:
            experiment_suffix: User-defined name. It will show up on data directory.
            n_pyloops: Number of repetition for running single sequence program. 
            
        Note from Zihao(03/03/2023):
        We call implement_parameters methods here instead of during init/load of DACManager, \
        because we want those modules/sequencers not being used to keep their default status.

        Note from Zihao(06/21/2023):
        I didn't use ** on these kwargs. I believe dict bring us better encapsulation.
        Pass in a whole dictionary can be more general than just pass 'something = somevalue'.
        It's because the key doen't need to be string anymore.

        Note from Zihao(09/09/2023):
        The first time one read this method may think some parts are overpackaging.
        It turns out after half year I end up at this form for a lot reasons.
        One reason is I want users to be able to run every line here without call self.run() itself.
        This will bring huge flexibility when running special/complicated experiments.
        """
        self.set_running_attributes(experiment_suffix, n_pyloops, process_kwargs, fitting_kwargs, plot_kwargs)
        self.make_sequence() 
        self.save_sequence()
        self.upload_sequence() 
        self.make_exp_dir()  # It also save a copy of yamls and jsons there.
        self.acquire_data()  # This is really run the thing and return to the IQ data in self.measurement.
        self.save_data()
        self.process_data()
        self.fit_data()
        self.save_data()
        self.plot()


    def check_attribute(self):
        """
        Check the qubits/resonators are always string with 'Q' or 'R'.
        Make sure each qubit has subspace and each resonator has level_to_fit.
        Make sure the pre_gate/post_gate is indeed in form of dictionary.
        Check total acquisition point in sequence program.
        Make sure classification is on when heralding is on.
        Note this method is usually called before we have self.tones and self.qudits.
        """
        for qudit in self.drive_qubits + self.readout_tones:
            if not qudit.startswith(('Q', 'R')): raise ValueError(f'The value of {qudit} is invalid.')
        
        assert hasattr(u, self.x_plot_unit), f'The plot unit {self.x_plot_unit} has not been defined.'
        if self.x_stop < self.x_start:
            raise ValueError('Please use ascending value for x_values.')
        if len(self.subspace) != len(self.drive_qubits):
            raise ValueError('Please specify subspace for each qubit.')
        if len(self.level_to_fit) != len(self.readout_resonators):
            raise ValueError('Please specify level_to_fit for each resonator.')
        if self.num_bins > 131072:
            raise ValueError('x_points * n_seqloops cannot exceed 131072! Please use n_pyloops!')
        if self.classification_enable < self.heralding_enable:
            raise ValueError('Please turn on classification for heralding.')


    @property
    def qudits(self):
        """
        As the first step towards dynamical attribute, we make self.qudits property.
        It allows user to modify self.tones of an instance without worry about the gates dataframe.
        Such modification usually happen after instantiation and before make_sequence.
        """
        return tone_to_qudit(self.tones)
    

    @property
    def readout_resonators(self):
        return tone_to_qudit(self.readout_tones)
    

    @property
    def rest_tones(self):
        return [tone for tone in self.tones if tone not in self.main_tones]


    @property
    def tones_(self):
        return [tone.replace('/', '_') for tone in self.tones]


    @property
    def main_tones_(self):
        return [main_tone.replace('/', '_') for main_tone in self.main_tones]
    

    @property
    def readout_tones_(self):
        return [readout_tone.replace('/', '_') for readout_tone in self.readout_tones]


    def make_tones_list(self):
        """
        Generate list attribute self.tones from existing attributes.
        It will have values like: ['Q3/01', 'Q3/12', 'Q3/23', 'Q4/01', 'Q4/12', 'R3/a', 'R4/a'].
        Each tone will map to a sequencer which can only run one frequency at a moment.
        By determine the tones, we actually determine the number of sequencer involved in experiment.
        self.main_tones will be the tones that those Rabi/Ramsey/DriveAmplitude happen on.
        self.rest_tones will be the tones that only do pre_gate/post_gate/readout_gate.

        As the result, Scan will have both static and dynamic attributes.
        Static attributes: (Include all user specified keyword arguments)
            drive_qubits, readout_tones, subspace, main_tones, level_to_fit, tones
        Dynamic attributes:
            qudits, readout_resonators, rest_tones, and all tones attributes end with underscroll.

        For more information, see self.make_sequence, DACManager.implement_parameters and start_sequencer.
        """
        self.tones = []

        # Determine tones list from self.subspace.
        for i, qubit in enumerate(self.drive_qubits):
            highest_level = split_subspace(self.subspace[i])[-1]

            for level in range(highest_level):
                self.tones.append(f'{qubit}/{level}{level+1}')

        # Add readout_tones
        self.tones += self.readout_tones

        # If main_tones keep default, generate it from self.subspace.
        if self.main_tones == []: self.main_tones = [f'{q}/{ss}' for q, ss in zip(self.drive_qubits, self.subspace)]


    def set_running_attributes(
            self,             
            experiment_suffix: str = '',
            n_pyloops: int = 1,
            process_kwargs: dict = None,
            fitting_kwargs: dict = None,
            plot_kwargs: dict = None):
        """
        Set input arguments as object attributes when calling self.run().
        It's useful when we overload self.run() method and still need to use these attributes.
        """
        self.experiment_suffix = experiment_suffix
        self.n_pyloops = n_pyloops
        self.n_reps = self.n_seqloops * self.n_pyloops

        self.process_kwargs = process_kwargs if process_kwargs is not None else {}
        self.fitting_kwargs = fitting_kwargs if fitting_kwargs is not None else {}
        self.plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
        self.attrs = {k: v for k, v in self.__dict__.items() if not k.startswith(('cfg', 'measurement'))}


    ##################################################
    def make_sequence(self):
        """
        Generate the self.sequences, which is a dictionary including all sequence \
        dictionaries that we will dump to json file.
        
        Example:
        self.sequences = {'Q3/01': Q3/01_sequence_dict,
                          'Q3/12': Q3/12_sequence_dict
                          'Q4/01': Q4/01_sequence_dict,
                          'Q4/12': Q4/12_sequence_dict
                          'R3': R3_sequence_dict,
                          'R4': R4_sequence_dict}
        
        Each sequence dictionary should looks like:
        sequence_dict = {'waveforms': waveforms,
                         'weights': weights,
                         'acquisitions': acquisitions,
                         'program': seq_prog}
        
        Please check the link below for detail:
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/basic_sequencing.html

        Note from Zihao (04/11/2023):
        From today, I decide to map each sequencer to each subspace rather than each qubit as previous code.
        This bring you flexibility on qutrit and qudit experiment, for example qutrit Rz gate.
        Please check the VariableManager and DACManager about varman['tones'] for more details.
        """
        # Gate and pulse dataframe for user to read/check gates.
        # Sequence will be correctly generated even without it.
        self.gate_df = dict_to_DataFrame(dic={}, name='', rows=self.qudits)
        self.pulse_df = dict_to_DataFrame(dic={}, name='', rows=self.tones)

        self.sequences = {tone: {} for tone in self.tones}        
        self.set_waveforms_acquisitions()
        
        self.init_program()
        self.start_loop()
        
        self.add_relaxation()
        if self.heralding_enable: self.add_heralding()
        self.add_gate(self.subspace_gate, 'Subspace')
        self.add_gate(self.pre_gate, 'Pregate')
        self.add_main()
        self.add_gate(self.post_gate, 'Postgate')
        self.add_readout()
        
        self.end_loop()
    
        
    def set_waveforms_acquisitions(self, add_special_waveforms: bool = True, **waveform_kwargs):
        """
        Generate waveforms, weights, acquisitions items in self.sequences[tone] dictionary.
        By default, we will add all special waveforms from GateManager.
        High waveform-demand scan like Rabi and Chevron can turn it to false.
        
        We skip weights acquisition, and I believe even if we need it,
        we can do it in post process. --Zihao(01/31/2023)
        Please check the link below for detail:
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/binned_acquisition.html
        """
        for tone in self.tones:
            if tone.startswith('Q'):
                length = self.qubit_pulse_length_ns
                shape = self.cfg.variables[f'{tone}/pulse_shape']

                waveforms = {'1qMAIN': {'data': get_waveform(length, shape, **waveform_kwargs), 
                                        'index': 0},
                             '1qDRAG': {'data': get_waveform(length, f'{shape}_derivative', **waveform_kwargs), 
                                        'index': 1}}
                weights = {}
                acquisitions = {}
            
            elif tone.startswith('R'):
                length = self.resonator_pulse_length_ns
                shape = self.cfg.variables[f'{tone}/pulse_shape']

                waveforms = {'RO': {'data': get_waveform(length, shape, **waveform_kwargs), 
                                    'index': 0}}
                weights = {}
                acquisitions = {'readout': {'num_bins': self.num_bins, 'index': 0}}
                if self.heralding_enable: acquisitions['heralding'] = {'num_bins': self.num_bins, 'index': 1}
            
            
            self.sequences[tone]['waveforms'] = waveforms
            self.sequences[tone]['weights'] = weights
            self.sequences[tone]['acquisitions'] = acquisitions

        if add_special_waveforms: self.add_special_waveforms()


    def add_special_waveforms(self):
        """
        Add all waveforms besides the single qubit gate and readout to the 'waveforms' in self.sequences.
        Remember the total waveform length for each sequencer cannot exceed 16384 data points.
        """
        for gate, gate_config in self.cfg.gates.manager_dict.items():
            for tone in self.tones:
                
                # We may not have defined such gate for all qubits/resonators.
                try:
                    pulse_dict = deepcopy(gate_config[tone])
                except KeyError:
                    continue

                # Pop out all necessary keys and others will become kwargs.
                length = round(pulse_dict.pop('length') * 1e9)
                shape = pulse_dict.pop('pulse_shape')

                waveforms = {f'{gate}MAIN': {'data': get_waveform(length, shape, **pulse_dict),
                                             'index': pulse_dict['waveform_index']},
                             f'{gate}DRAG': {'data': get_waveform(length, shape+'_derivative', **pulse_dict),
                                             'index': pulse_dict['waveform_index'] + 1}}

                self.sequences[tone]['waveforms'].update(waveforms)

        
    def init_program(self):
        """
        Create sequence program and initialize some built-in registers.
        Please do not change the convention here since their function have been hardcoded later.
        """
        for tone in self.tones:
            program = f"""
        # R0 count n_seqloops, descending.
        # R1 count bin for acquisition, ascending.
        # R2 qubit relaxation time in microseconds, descending.
        # R3 count x_points, descending.
        # R4 is specific x_value.
        # R5 count y_points, descending.
        # R6 is specific y_value.
        # R7 R8 R9 R10 are left for backup.
        # Other register up to R63 can be used freely.
        
                    wait_sync        8
                    move             {self.n_seqloops},R0
                    move             0,R1
        """
            self.sequences[tone]['program'] = program
            
            
    def start_loop(self):
        """
        Add the head of loop structure to sequence program.
        We assume sequence loop is always the outermost loop.
        For each inner loop, either x loop or y loop in future, \
        we need to assign initial value before entering the loop.
        """
        self.add_seqloop()
        self.add_xinit()
        self.add_xloop()
        self.add_sequence_start()
        
     
    def add_seqloop(self):
        """
        Add seq_loop to sequence program.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """
        seq_loop:   
        """
        
     
    def add_xinit(self):
        """
        Set necessary initial value of x parameter to the registers, especially R3 & R4. 
        Child class can super this method to add more initial values.
        """
        for tone in self.tones: self.sequences[tone]['program'] += f"""
                    move             {self.x_points},R3    
        """
        
        
    def add_xloop(self):
        """
        Add x_loop to sequence program.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """            
        xpt_loop: 
        """
        
        
    def add_sequence_start(self):
        """
        Add sync and phase resetting instruction to sequence program.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """     
                #-----------Start-----------
                    wait_sync        8               # Sync at beginning of the loop.
                    reset_ph                         # Reset phase to eliminate effect of previous VZ gate.
                    set_mrk          15              # Enable all markers (binary 1111) for switching on output.
                    upd_param        8               # Update parameters and wait 8ns.
        """
            
        
    def add_relaxation(self, label: str | int = ''):
        """
        Add relaxation to sequence program.
        Allow an optional label to avoid label repeat in Q1ASM.
        
        Note from Zihao(03/31/2023):
        Although we can use add_wait to achieve same effect,
        I don't want add a bunch of nonsense 'I' to my gate_df.  
        """
        relaxation_time_s = self.cfg.variables['common/relaxation_time']
        relaxation_time_us = int( np.ceil(relaxation_time_s*1e6) )
        relaxation = f"""
                #-----------Relaxation-----------
                    move             {relaxation_time_us},R2
        rlx_loop{label}:   wait             1000
                    loop             R2,@rlx_loop{label}
        """
        for tone in self.tones: self.sequences[tone]['program'] += relaxation
        
        
    def add_heralding(self, name: str = 'Heralding', add_label: bool = True, 
                      concat_df: bool = True, acq_index: int = 1, delay: float = None):
        """
        Add heralding with short delay(relaxation) to the sequence program.
        """
        self.add_readout(name=name, add_label=add_label, concat_df=concat_df, acq_index=acq_index)
        delay = self.cfg.variables['common/heralding_delay'] if delay is None else delay
        tof = self.cfg.variables['common/tof']
        self.add_wait(name=name+'Delay', length=round((delay-tof) / u.ns), add_label=add_label, concat_df=concat_df)
        
        
    def add_main(self):        
        """
        Add main content of the parameter sweep. 
        Suppose to be called by child class.
        """
        print('Scan: The base experiment class has been called. No main sequence will be added.')

    
    def add_readout(self, name: str = 'Readout', add_label: bool = True, 
                    concat_df: bool = True, acq_index: int = 0):
        """
        Add readout/heralding with acquisition to sequence program.
        """
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        self.add_gate(gate=self.readout_gate, lengths=[tof_ns + self.resonator_pulse_length_ns],
                      name=name, add_label=add_label, concat_df=concat_df, acq_index=acq_index)
        
        
    def end_loop(self):
        """
        End all loops and add stop to sequence program.
        
        Note from Zihao(03/07/2023):
        In principle, we need to add x_step to the register that store x_value before ending x loop.
        Here I absorb that part into add_main for continuity and readability.
        For multidimensional scan, it only works when x loop is the innermost loop.
        Fortunately, this is indeed the most efficient way to reuse code of 1D scan onto 2D scan.
        If we make it separate, all child class need to modify it and become not elegant.
        """
        self.add_sequence_end()
        self.end_xloop()
        self.end_seqloop()
            
        
    def add_sequence_end(self):
        """
        Count next acquisition bin (R1) and turn off all output.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """
                #-----------Stop-----------
                    add              R1,1,R1
                    set_mrk          0               # Disable all markers (binary 0000) for switching off output.
                    upd_param        8               # Update parameters and wait 4ns.     
        """
        

    def end_xloop(self):
        """
        Add end of x loop.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """
                    loop             R3,@xpt_loop         
        """
        
        
    def end_seqloop(self):
        """
        End sequence loop and stop the sequence program.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """
                    loop             R0,@seq_loop
                    
                    stop             
        """


    def add_wait(self, name: str, length: int, add_label: bool = True,
                 concat_df: bool = True, divisor_ns: int = 65528):
        """
        Add wait/Identity to sequence program with user-defined length in [ns].
        The maximum time of 'wait' instruction is 65535, so I divide it into multiple of divisor \
        along with a remainder, and use multiple instruction to achieve it.
        Here I treat it as a gate, which means we can add label if we want, \
        and we can see it in self.gate_df
        """
        if length < 4: raise ValueError(f'The wait time need to be at least 4ns. Now it is {length}.')
        multiple = round(length // divisor_ns)
        remainder = round(length % divisor_ns)
        
        gate = {qudit: ['I' for _ in range(multiple+1)] for qudit in self.qudits}
        lengths = [divisor_ns for _ in range(multiple)] + [remainder]
        self.add_gate(gate, name, lengths, add_label=add_label, concat_df=concat_df)
    
            
    def add_gate(self, gate: dict[str: list[str]], name: str, lengths: list[int] = None,
                 add_label: bool = True, concat_df: bool = True, **pulse_kwargs):
        """
        The general method for adding gates to sequence.
        We will generate the Pandas DataFrame of pre_gate, post_gate, readout, with padded 'I'.
        All qubits and resonators will become the (row) index of dataframe.
        An additional interger attribute 'length' in [ns] will be associated with each column.
        If lengths is shorter than number of gate, it will be padded using the last length.
        Please remember all labels created in Q1ASM should have different names.
        
        Attributes:
            gate: {'Q0': ['X180_01'], 'Q1': ['X90_12', 'Y90_12']}.
            lengths: The duration of each gate(column) in [ns]. Default is single qubit gate time.
            name: String in sequence program to improve readability.
            
        Example of a subspace_gate + pre_gate DataFrame:
           subspace_0 subspace_1 Pregate_0 Pregate_1
        Q3    X180_01          I    Y90_01         I
        Q4    X180_01    X180_12    Y90_01    Z90_12
        R3          I          I         I         I
        R4          I          I         I         I

        Note from Zihao(04/11/2023):
        The each gate will then be decomposed to one or more pulses on different sequencer.
        Then pulse_interpreter will make them into Q1ASM string and add them to sequence program.
        The core difference between gate and pulse here is gate usually map to qubit/resonator, \
        whereas pulse has to map to each tone, which is each sequencer on Qblox.
        For example, a qutrit hadamard need two tones multiplexing which requires two sequencers to do that.
        In such case, gate_df still have a 'H3' on one qutrit 'Q3' with a gate length.
        However, pulse_df will have 'H3_01' in 'Q3/01' and 'H3_12' in 'Q3/12'.
        These two pulses are in same column but different rows, so same moment but different sequencers.
        """
        gate_df = dict_to_DataFrame(gate, name, self.qudits)  # Each row is a qudit.

        default_lengths = [self.qubit_pulse_length_ns for _ in range(gate_df.shape[1])]
        lengths = make_it_list(lengths, default_lengths) 
        if len(lengths) != gate_df.shape[1]: raise ValueError(f'Scan: Please specify length for all gates(columns)!')

        pulse_df = gate_transpiler(gate_df, self.tones)  # Rightnow it gives same number of columns as gate_df.
        self.add_pulse(pulse_df, pulse_lengths=lengths, add_label=add_label, **pulse_kwargs)
        
        # Concatenate a larger dataframe while keeping length for user to read/check it.
        if concat_df: 
            self.gate_df = pd.concat([self.gate_df, gate_df], axis=1)
            self.pulse_df = pd.concat([self.pulse_df, pulse_df], axis=1)
            
            
    def add_pulse(self, pulse_df: pd.DataFrame, pulse_lengths: list[int], add_label: bool = True, **pulse_kwargs):
        """
        Interpret the pulse dataframe to string and add it to the sequence program of each sequencer.
        Here we assume user has specified length of each column.
        """
        for i, col_name in enumerate(pulse_df):
            column = pulse_df[col_name]
            length = pulse_lengths[i]
                
            for tone in self.tones:
                pulse_prog = '' if not add_label else f"""
                # -----------{col_name}-----------
        {col_name}:  """
        
                pulse_prog += pulse_interpreter(cfg=self.cfg, tone=tone, pulse_string=column[tone], 
                                                length=length, **pulse_kwargs)
                self.sequences[tone]['program'] += pulse_prog
        
        
    ##################################################    
    def save_sequence(self, jsons_path: str = None):
        """
        Create json file of sequence for each sequencer/qudit and save it.
        Allow user to pass a path of directory to save jsons at another place.
        A text file of sequence program will also be saved for reading.
        """
        if jsons_path is None:
            jsons_path = self.jsons_path 

        for tone, sequence_dict in self.sequences.items():
            tone_ = tone.replace('/', '_')
            file_path = os.path.join(jsons_path, f'{tone_}_sequence.json')
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(sequence_dict, file, indent=4)
                
            txt_file_path = os.path.join(jsons_path, f'{tone_}_sequence_program.txt')
            with open(txt_file_path, 'w', encoding='utf-8') as txt:
                txt.write(sequence_dict['program'])


    def upload_sequence(self):
        """
        Setup the hardware instrument and upload json files of sequence to the instrument.
        """
        self.cfg.DAC.implement_parameters(self.tones, self.jsons_path) 
            
    
    def make_exp_dir(self):
        """
        Create an experiment directory under cfg.data.base_directory.
        Then save a copy of jsons and yamls to experiment directory.
        
        Note from Zihao(02/14/2023):
        If sequence program has problem, then self.upload_sequence() will raise Error.
        In that case we won't create the junk experiment folder.
        If problem happen after we start_sequencer, then it worth to create the experiment folder.
        Because we may already get some data and want to save it by manually calling save_data().
        """
        self.cfg.data.make_exp_dir(experiment_type='_'.join([*self.main_tones_, self.scan_name]),
                                   experiment_suffix=self.experiment_suffix)
        
        self.data_path = self.cfg.data.data_path
        self.datetime_stamp = self.cfg.data.datetime_stamp
        
        self.cfg.save(yamls_path=self.cfg.data.yamls_path, verbose=False)
        self.save_sequence(jsons_path=self.cfg.data.jsons_path)
        
        for rt_ in self.readout_tones_: os.makedirs(os.path.join(self.data_path, 'IQplots', rt_))
    
    
    def acquire_data(self, keep_raw: bool = False):
        """
        Create measurement dictionary, then start sequencer and save data into this dictionary.
        self.measurement should only have resonators' names as keys.
        Inside each resonator should be its subtones and consistent name of processing.
        After all loops, the 'Heterodyned_readout' usually has shape (2, n_pyloops, n_seqloops*x_points).
        We will reshape it to (2, n_reps, x_points) later by ProcessManager, where n_reps = n_seqloops * n_pyloops.

        Note from Zihao(07/22/2024):
        I use the fact that from Python 3.7 and later, dictionary is ordered.
        This helps us keep a consistent order of the readout tones.
        As long as the users use same order between CalibrateClassification and other experiments, \
        they will have correct multitone GMM processing result.
        """
        self.measurement = {rr: {} for rr in self.readout_resonators}

        for rt in self.readout_tones:
            meas_sub = {}
            for key in self.sequences[rt]['acquisitions'].keys():
                meas_sub[f'Heterodyned_{key}'] = [[],[]]
                if keep_raw: meas_sub[f'raw_{key}'] = [[],[]]

            rr, subtone = rt.split('/')
            self.measurement[rr][subtone] = meas_sub
        
        print('Scan: Start sequencer.')
        for i in range(self.n_pyloops):
            self.cfg.DAC.start_sequencer(self.tones, self.readout_tones, self.measurement, self.jsons_path, keep_raw)
            print(f'Scan: Pyloop {i} finished!')

        self.n_runs += 1
        self.measurements.append(self.measurement)


    def save_data(self):
        """
        Save the measurement dictionary along with attributes of the Scan to measurement.hdf5 under self.data_path.
        """
        self.cfg.data.save_measurement(self.data_path, self.measurement, self.attrs)

            
    def process_data(self):
        """
        Process the data by performing reshape, rotation, average, GMM, fit, plot, etc.
        We keep this layer here to provide possibility to inject functionality between acquire and fit.
        For example, in 2D scan, we need to give it another shape.
        """
        self.cfg.process.process_data(self.measurement, 
                                      shape=(2, self.n_reps, self.x_points),
                                      process_kwargs=self.process_kwargs)
        

    def fit_data(self, x: list | np.ndarray = None, **fitting_kwargs): 
        """
        Fit data in measurement dictionary and save result back into it.
        We will also keep fit result as attribute self.fit_result for future use.
        The model should be better picked from qtrlb.processing.fitting.
        data_dict['to_fit'] usually have shape (n_levels, x_points), or (2, x_points) without classification.
        Allow multi-dimensional fitting through fitting_kwargs by passing 'y' or even 'z'.
        
        Note from Zihao(03/17/2023):
        I leave an interface for x because it's convenient to fit multidimensional scan as 1D scan.
        In that case we can pass whichever axis as horizontal axis here.
        The only cost is to process self.measurement[r]['to_fit'] to correct shape.
        """
        self.fit_result = {rr: None for rr in self.readout_resonators}
        if self.fitmodel is None: return
        if x is None: x = self.x_values
        
        for i, rr in enumerate(self.readout_resonators):
            try:
                level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']
                self.fit_result[rr] = fit(input_data=self.measurement[rr]['to_fit'][level_index],
                                          x=x, fitmodel=self.fitmodel, **fitting_kwargs)
                
                params = {v.name:{'value':v.value, 'stderr':v.stderr} for v in self.fit_result[rr].params.values()}
                self.measurement[rr]['fit_result'] = params
                self.measurement[rr]['fit_model'] = str(self.fit_result[rr].model)
            except Exception:
                self.fitting_traceback = traceback.format_exc()  # Return a string to debug.
                print(f'Scan: Failed to fit {rr} data. See traceback by print Scan.fitting_traceback.')
                self.measurement[rr]['fit_result'] = None
                self.measurement[rr]['fit_model'] = str(self.fitmodel)
                

    def plot(self):
        """
        Plot the measurement result and save figures into experiment directory.
        """
        try:
            self.plot_main()
            if self.cfg.DAC.test_mode: return
            if self.classification_enable: self.plot_populations()
            if self.cfg['variables.common/plot_IQ'] is True: self.plot_IQ()

        except Exception:
            self.plotting_traceback = traceback.format_exc()
            print(f'Scan: Failed to plot {self.datetime_stamp} data. See Scan.plotting_traceback.')
            self.figures = {rr: None for rr in self.readout_resonators}


    def plot_main(self, text_loc: str = 'lower right', dpi: int = 150):
        """
        Plot the main result along with fitting, if not failed.
        Figure will be saved to data directory and show up on python console.
        
        Reference about add_artist method of Axes object:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_artist.html
        """
        self.figures = {}
        
        for i, rr in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']      
            title = f'{self.datetime_stamp}, {self.scan_name}, {rr}'
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            if self.classification_enable:
                ylabel = fr'$P_{{\left|{self.level_to_fit[i]}\right\rangle}}$'
            else:
                ylabel = 'I-Q Coordinate (Rotated) [a.u.]'
            
            fig, ax = plt.subplots(1, 1, dpi=dpi)
            ax.plot(self.x_values / self.x_unit_value, self.measurement[rr]['to_fit'][level_index], 'k.')
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
            
            if self.fit_result[rr] is not None: 
                # Raise resolution of fit result for smooth plot.
                x = np.linspace(self.x_start, self.x_stop, self.x_points * 3)  
                y = self.fit_result[rr].eval(x=x)
                ax.plot(x / self.x_unit_value, y, 'm-')
                
                # AnchoredText stolen from Ray's code.
                fit_text = '\n'.join([f'{v.name} = {v.value:0.5g}' for v in self.fit_result[rr].params.values()])
                anchored_text = AnchoredText(fit_text, loc=text_loc, prop={'color':'m'})
                ax.add_artist(anchored_text)

            fig.savefig(os.path.join(self.data_path, f'{rr}.png'))
            self.figures[rr] = fig
            
            
    def plot_IQ(self, 
                IQ_key: str = 'IQrotated_readout', 
                c_key: str = 'GMMpredicted_readout', 
                mask_key: str = 'Mask_heralding',
                dpi: int = 75):
        """
        Plot the IQ point for each element in self.x_values.
        Figure will be saved to data directory without show up on python console.
        Additionally, if we enable classification, we use GMM predicted result as colors.
        Furthermore, if we enable heralding, we will also make plot for those data passed heralding test. 
        
        Note from Zihao(03/02/2023):
        Only when somebody really try to write this code do they realize how difficult it is.
        With for-for-if three layers nested, this is the best I can do.
        The heralding enable is protected since we have self.check_attribute().
        """
        for rt_ in self.readout_tones_:
            rr, subtone = rt_.split('_')
            Is, Qs = self.measurement[rr][subtone][IQ_key]
            left, right = (np.min(Is), np.max(Is))
            bottom, top = (np.min(Qs), np.max(Qs))

            for x in range(self.x_points):
                I = Is[:,x]
                Q = Qs[:,x]
                c = self.measurement[rr][c_key][:,x] if self.classification_enable else None
                fig, ax = plt.subplots(1, 1, dpi=dpi)
                ax = plot_IQ(ax, I, Q, c, title=f'{x}', xlim=(left, right), ylim=(bottom, top))
                fig.savefig(os.path.join(self.data_path, 'IQplots', rt_, f'{x}.png'))
                fig.clear()
                plt.close(fig)
                
                if self.heralding_enable:
                    mask = self.measurement[rr][mask_key][:,x] 
                    I_masked = np.ma.MaskedArray(I, mask=mask)
                    Q_masked = np.ma.MaskedArray(Q, mask=mask)
                    c_masked = np.ma.MaskedArray(c, mask=mask)
                    fig, ax = plt.subplots(1, 1, dpi=dpi)
                    ax = plot_IQ(ax, I_masked, Q_masked, c_masked, 
                                 title=f'{x}', xlim=(left, right), ylim=(bottom, top))
                    fig.savefig(os.path.join(self.data_path, 'IQplots', rt_, f'heralded_{x}.png'))
                    fig.clear()
                    plt.close(fig)


    def plot_populations(self, dpi: int = 150):
        """
        Plot populations for all levels, both with and without readout correction.
        """           
        for rr in self.readout_resonators:
            fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=dpi)
            for i, level in enumerate(self.cfg[f'variables.{rr}/readout_levels']):
                ax[0].plot(self.x_values / self.x_unit_value, self.measurement[rr]['PopulationNormalized_readout'][i], 
                           c=self.color_list[level], ls='-', marker='.', label=fr'$P_{{{level}}}$')
                ax[1].plot(self.x_values / self.x_unit_value, self.measurement[rr]['PopulationCorrected_readout'][i], 
                           c=self.color_list[level], ls='-', marker='.', label=fr'$P_{{{level}}}$')

            xlabel = f'{self.x_plot_label}[{self.x_plot_unit}]'
            ax[0].set(xlabel=xlabel, ylabel='Uncorrected populations', ylim=(-0.05, 1.05))
            ax[1].set(xlabel=xlabel, ylabel='Corrected populations', ylim=(-0.05, 1.05))
            ax[0].legend()
            ax[1].legend()
            ax[0].set_title(f'{self.datetime_stamp}, {self.scan_name}, {rr}')
            fig.savefig(os.path.join(self.data_path, f'{rr}_Population.png'))
            fig.clear()
            plt.close(fig)


    def switch_level(self, level_to_fit: int | list[int]):
        """ 
        A covenient method for change level_to_fit then redo fit and plot.
        """
        self.level_to_fit = make_it_list(level_to_fit)
        if len(self.level_to_fit) != len(self.readout_resonators):
            raise ValueError('Please specify level for all resonators.')
        self.fit_data()
        self.plot_main()


    def normalize_subspace_population(self, subspace: str | list[str] = None, dpi: int = 150):
        """
        Allow user to fit and plot only the population inside main subspace to compensate overall decay.
        It will overwrite the self.fit_result and self.figures, but not the saved plot and fit_result in hdf5.
        """
        subspace = make_it_list(subspace, self.subspace)
        if not self.classification_enable: 
            raise ValueError('This function only work when enabling classification.') 
        if len(subspace) != len(self.readout_resonators): 
            raise ValueError('Please specify fitting subspace for each resonator.')

        for i, rr in enumerate(self.readout_resonators):
            # Normalization.
            level_low, level_high = split_subspace(subspace[i])
            P_low = self.measurement[rr]['to_fit'][level_low - self.cfg[f'variables.{rr}/lowest_readout_levels']]
            P_high = self.measurement[rr]['to_fit'][level_high - self.cfg[f'variables.{rr}/lowest_readout_levels']]
            self.measurement[rr]['to_fit_SubspaceNormalized'] = P_high / (P_high + P_low)

            # Fitting
            self.fit_result[rr] = fit(self.measurement[rr]['to_fit_SubspaceNormalized'], self.x_values, self.fitmodel)

            params = {v.name:{'value':v.value, 'stderr':v.stderr} for v in self.fit_result[rr].params.values()}
            self.measurement[rr]['fit_result_SubspaceNormalized'] = params
            self.save_data()

            # Plot    
            fig, ax = plt.subplots(1, 1, dpi=dpi)
            ax.plot(self.x_values / self.x_unit_value, self.measurement[rr]['to_fit_SubspaceNormalized'], 'k.')
            ax.set(xlabel=self.x_plot_label + f'[{self.x_plot_unit}]', 
                   ylabel=fr'$\frac{{P_{{{level_high}}}}}{{P_{{{level_high}}} + P_{{{level_low}}}}}$',
                   title=f'{self.datetime_stamp}, {self.scan_name}, {rr}')
            
            x3 = np.linspace(self.x_start, self.x_stop, self.x_points * 3)  
            ax.plot(x3 / self.x_unit_value, self.fit_result[rr].eval(x=x3), 'm-')
            fit_text = '\n'.join([f'{v.name} = {v.value:0.5g}' for v in self.fit_result[rr].params.values()])
            ax.add_artist(AnchoredText(fit_text, loc='upper right', prop={'color':'m'}))

            fig.savefig(os.path.join(self.data_path, f'{rr}_SubspaceNormalized.png'))
            self.figures[rr] = fig
    
    
    @staticmethod
    def frequency_translator(freq: float, freq_step: float = 0.25, bit: int = 32):
        """
        Because qblox use 32 bit register to help run sequence program,
        The value we assign to it must be non-negative integer [0, 2**32).
        This integer will be translated by instrument into binary without consider two's complement.
        Then it will be interpreted using two's complement to realize negative value.
        For frequency, where the integer represents multiples of 0.25Hz, \
        the register can store frequency between [-2**29, 2**29) Hz.
        This function helps to calculate that non-negative integer based on a input frequency in [Hz].
        
        Note from Zihao(03/14/2023):
        This is useful when we assign the frequency to register and treat it as a variable.
        When we call set_freq instruction in Q1ASM program, we can just pass negative frequency * 4.
        """
        if not -500e6 <= freq <= 500e6: raise ValueError('The frequency must between +-500MHz.')
        freq_4 = round(freq / freq_step)
        
        twos_complement_binary_str = format(freq_4 if freq_4 >= 0 else (1 << bit) + freq_4, f'0{bit}b')
        return int(twos_complement_binary_str, 2)
    

    @staticmethod
    def gain_translator(gain: float, gain_resolution: int = 32768, bit: int = 32):
        """
        This is the similar case as frequence_translator.
        The gain passed in here take value [-1.0, 1.0).
        We need to make it integer between [-32768, 32768) for set_awg_gain instruction.
        Still, this is useful when we assign the gain to register and treat it as a variable.
        When we call set_awg_gain with Immediate in Q1ASM program, we can just pass negative gain * 32768.
        """
        if not -1 <= gain < 1: raise ValueError('The gain must between [-1.0, 1.0).')
        gain = round(gain * gain_resolution)
        
        twos_complement_binary_str = format(gain if gain >= 0 else (1 << bit) + gain, f'0{bit}b')
        return int(twos_complement_binary_str, 2)




class Scan2D(Scan):
    """ Base class for all 2D parameter-sweep experiment.
        It's a small extension added on the Scan class.
        In sequence, we will sweep over x value first, then y, then repeat.
        The framework of process and fit of 1D Scan still work here.
        
        Note from Zihao(03/06/2023):
        I don't use super().__init__ here since I don't want to allow dependency injection.
        It is because of the inheritance structure of those child class of Scan2D.
        For instance, ChevronScan inherit both Scan2D and RabiScan, \
        but we don't want to run RabiScan.__init__() since it will redefine out x_axis.
    """
    def __init__(self, 
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 scan_name: str,
                 x_plot_label: str, 
                 x_plot_unit: str, 
                 x_start: float, 
                 x_stop: float, 
                 x_points: int, 
                 y_plot_label: str,
                 y_plot_unit: str,
                 y_start: float,
                 y_stop: float,
                 y_points: int,
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None,
                 **attr_kwargs):
        
        Scan.__init__(self,
                      cfg=cfg, 
                      drive_qubits=drive_qubits,
                      readout_tones=readout_tones,
                      scan_name=scan_name,
                      x_plot_label=x_plot_label, 
                      x_plot_unit=x_plot_unit, 
                      x_start=x_start, 
                      x_stop=x_stop, 
                      x_points=x_points, 
                      subspace=subspace,
                      main_tones=main_tones,
                      pre_gate=pre_gate,
                      post_gate=post_gate,
                      n_seqloops=n_seqloops,
                      level_to_fit=level_to_fit,
                      fitmodel=fitmodel,
                      y_plot_label=y_plot_label,
                      y_plot_unit=y_plot_unit,
                      y_start=y_start,
                      y_stop=y_stop,
                      y_points=y_points,
                      num_bins=n_seqloops * x_points * y_points,
                      **attr_kwargs)
         
        self.y_values = np.linspace(self.y_start, self.y_stop, self.y_points)
        self.y_step = (self.y_stop - self.y_start) / (self.y_points-1) if self.y_points != 1 else 0
        self.y_unit_value = getattr(u, self.y_plot_unit)
    

    def check_attribute(self):
        """
        Check the attributes for 2D Scan.
        """
        super().check_attribute()
        assert hasattr(u, self.y_plot_unit), f'The plot unit {self.y_plot_unit} has not been defined.'
        if self.y_stop < self.y_start:
            raise ValueError('Please use ascending value for y_values.')
        if self.num_bins > 131072:
            raise ValueError('x_points * y_points * n_seqloops cannot exceed 131072! Please use n_pyloops!')
    
    
    def start_loop(self):
        """
        Sequence loop is the outermost loop, then y loop, and x loop is innermost.
        """
        self.add_seqloop()
        self.add_yinit()
        self.add_yloop()
        self.add_xinit()
        self.add_xloop()
        self.add_sequence_start()


    def add_yinit(self):
        """
        Set necessary initial value of y parameter to the registers, especially R5 & R6. 
        Child class can super this method to add more initial values.
        """
        for tone in self.tones: self.sequences[tone]['program'] += f"""            
                    move             {self.y_points},R5    """


    def add_yloop(self):
        """
        Add y_loop to sequence program.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """            
        ypt_loop:   """


    def end_loop(self):
        """
        End all loops and add stop to sequence program. 
        See parent class for notes of this method.
        """
        self.add_sequence_end()
        self.end_xloop()
        self.add_yvalue()
        self.end_yloop()
        self.end_seqloop()


    def add_yvalue(self):
        """
        Change value of the register that represent y_value (R6).
        It need to be outside x_loop, but inside y_loop.
        """
        print('Scan2D: The base experiment class has been called. y_value (R6) will not change during loop.')
        

    def end_yloop(self):
        """
        Add end of x loop.
        """
        y_end = """
                    loop             R5,@ypt_loop    """
        for tone in self.tones: self.sequences[tone]['program'] += y_end


    def process_data(self):
        """
        Process the data by performing reshape, rotation, average, GMM, fit, plot, etc.
        We keep this layer here to provide possibility to inject functionality between acquire and fit.
        """
        self.cfg.process.process_data(self.measurement, 
                                      shape=(2, self.n_reps, self.y_points, self.x_points),
                                      process_kwargs=self.process_kwargs)


    def plot(self):
        """
        Note from Zihao(11/15/2023):
        I haven't add plot_population here since there is too many plots and hard to manage.
        The IQ plots are also a lot, but we can turn it off in yaml easily.
        Sometimes I have to do it quickly. I'm sorry.
        """
        try:
            self.plot_main()
            if self.cfg.DAC.test_mode: return
            if self.cfg['variables.common/plot_IQ'] is True: self.plot_IQ()
        except Exception:
            self.plotting_traceback = traceback.format_exc()
            print(f'Scan2D: Failed to plot {self.datetime_stamp} data. See Scan2D.plotting_traceback.')
            self.figures = {rr: None for rr in self.readout_resonators}


    def plot_main(self, dpi: int = 150):
        """
        Plot population for all levels. 
        If we disable classification, plot both quadrature.
        """
        self.figures = {}

        for rr in self.readout_resonators:
            data = self.measurement[rr]['to_fit']
            n_subplots = len(data)
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            ylabel = self.y_plot_label + f'[{self.y_plot_unit}]'
            title = f'{self.datetime_stamp}, {self.scan_name}, {rr}'
            
            fig, ax = plt.subplots(1, n_subplots, figsize=(7 * n_subplots, 8), dpi=dpi)

            for l in range(n_subplots):
                level = self.cfg[f'variables.{rr}/readout_levels'][l]
                this_title = title + fr', $P_{{{level}}}$' if self.classification_enable else title
                
                image = ax[l].imshow(data[l], cmap='RdBu_r', interpolation='none', aspect='auto', 
                                     origin='lower', extent=[np.min(self.x_values) / self.x_unit_value, 
                                                             np.max(self.x_values) / self.x_unit_value, 
                                                             np.min(self.y_values) / self.y_unit_value, 
                                                             np.max(self.y_values) / self.y_unit_value])
                ax[l].set(title=this_title, xlabel=xlabel, ylabel=ylabel)
                fig.colorbar(image, ax=ax[l], label='Probability/Coordinate', location='top')
                
            fig.savefig(os.path.join(self.data_path, f'{rr}.png'))
            self.figures[rr] = fig


    def plot_IQ(self, 
                IQ_key: str = 'IQrotated_readout', 
                c_key: str = 'GMMpredicted_readout', 
                mask_key: str = 'Mask_heralding',
                dpi: int = 75):
        """
        Plot the IQ point for each element in self.x_values and ALSO self.y_values.
        For 2D scan, it will usually generate a lot of IQ plot. So it's not recommended.
        For more information, please check Scan.plot_IQ()
        The only difference here is we have double for loop, and we need to index y in data and plot.
        """
        for rt_ in self.readout_tones_:
            rr, subtone = rt_.split('_')
            Is, Qs = self.measurement[rr][subtone][IQ_key]
            left, right = (np.min(Is), np.max(Is))
            bottom, top = (np.min(Qs), np.max(Qs))

            for y in range(self.y_points):
                for x in range(self.x_points):
                    I = Is[:,y,x]
                    Q = Qs[:,y,x]
                    c = self.measurement[rr][c_key][:,y,x] if self.classification_enable else None
                    fig, ax = plt.subplots(1, 1, dpi=dpi)
                    ax = plot_IQ(ax, I, Q, c, title=f'y{y}_x{x}', xlim=(left, right), ylim=(bottom, top))
                    fig.savefig(os.path.join(self.data_path, 'IQplots', rt_, f'y{y}_x{x}.png'))
                    fig.clear()
                    plt.close(fig)
                    
                    if self.heralding_enable:
                        mask = self.measurement[rr][mask_key][:,x] 
                        I_masked = np.ma.MaskedArray(I, mask=mask)
                        Q_masked = np.ma.MaskedArray(Q, mask=mask)
                        c_masked = np.ma.MaskedArray(c, mask=mask)
                        fig, ax = plt.subplots(1, 1, dpi=dpi)
                        ax = plot_IQ(ax, I_masked, Q_masked, c_masked,
                                     title=f'y{y}_x{x}', xlim=(left, right), ylim=(bottom, top))
                        fig.savefig(os.path.join(self.data_path, 'IQplots', rt_, f'heralded_y{y}_x{x}.png'))
                        fig.clear()
                        plt.close(fig)

