import os
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qtrlb.utils.units as u
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.offsetbox import AnchoredText
from lmfit import Model
from qtrlb.utils.waveforms import get_waveform
from qtrlb.utils.pulses import pulse_interpreter
from qtrlb.processing.fitting import fit




class Scan:
    """ Base class for all parameter-sweep experiment.
        The framework of how experiment flow will be constructed here.
        It should be used as parent class of specific scan rather than being instantiated directly.
        In __init__(), we shape the input parameter to better structure.
        We will check those attribute to ensure they have correct shape/values.
        
        Attributes:
            cfg: A MetaManager.
            drive_qubits: 'Q2', or ['Q3', 'Q4'].
            readout_resonators: 'R3' or ['R1', 'R5'].
            scan_name: 'drive_amplitude', 't1', 'ramsey'.
            x_start: 0.
            x_stop: 320e-9.
            x_points: number of points on x_axis. Start and stop points will be both included.
            subspace: '12' or ['01', '12'], length should be same as drive_qubits.
            prepulse: {'Q0': ['X180_01'], 'Q1': ['X90_12', 'Y90_12']}
            postpulse: Same requirement as prepulse.
            n_seqloops: Number of repetition inside sequence program. Total repetition will be n_seqloops * n_pyloops.
            level_to_fit: 0, 1 or [0,1,0,0], length should be same as readout_resonators.
            fitmodel: It should be better to pick from qtrlb.processing.fitting.
    """
    def __init__(self, 
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 scan_name: str,
                 x_plot_label: str, 
                 x_plot_unit: str, 
                 x_start: float, 
                 x_stop: float, 
                 x_points: int, 
                 subspace: str | list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = None):
        self.cfg = cfg
        self.drive_qubits = self.make_it_list(drive_qubits)
        self.readout_resonators = self.make_it_list(readout_resonators)
        self.scan_name = scan_name
        self.x_plot_label = x_plot_label
        self.x_plot_unit = x_plot_unit
        self.x_start = x_start
        self.x_stop = x_stop
        self.x_points = x_points
        self.subspace = self.make_it_list(subspace) if subspace is not None else ['01']*len(self.drive_qubits)
        self.prepulse = prepulse if prepulse is not None else {}
        self.postpulse = postpulse if postpulse is not None else {}
        self.n_seqloops = n_seqloops
        self.level_to_fit = self.make_it_list(level_to_fit) if level_to_fit is not None else [0]*len(self.readout_resonators)
        self.fitmodel = fitmodel
        
        self.n_runs = 0
        self.measurements = []
        self.qudits = self.drive_qubits + self.readout_resonators
        self.classification_enable = self.cfg.variables['common/classification']
        self.heralding_enable = self.cfg.variables['common/heralding']
        self.x_values = np.linspace(self.x_start, self.x_stop, self.x_points)
        self.x_step = (self.x_stop - self.x_start) / (self.x_points-1) if self.x_points != 1 else 0 
        self.num_bins = self.n_seqloops * self.x_points
        self.jsons_path=os.path.join(self.cfg.working_dir, 'Jsons')
        
        self.check_attribute()
        
        self.x_unit_value = getattr(u, self.x_plot_unit)
        self.subspace_pulse = {q: [f'X180_{l}{l+1}' for l in range(int(ss[0]))] \
                               for q, ss in zip(self.drive_qubits, self.subspace)}
        self.readout_pulse = {r: ['RO'] for r in self.readout_resonators}
        

        
    def run(self, 
            experiment_suffix: str = '',
            n_pyloops: int = 1):
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
        """
        self.experiment_suffix = experiment_suffix
        self.n_pyloops = n_pyloops
        self.n_reps = self.n_seqloops * self.n_pyloops
        self.attrs = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith('_')}
        
        self.make_sequence() 
        self.save_sequence()
        self.cfg.DAC.implement_parameters(self.drive_qubits, self.readout_resonators, self.jsons_path) 
        self.make_exp_dir()  # It also save a copy of yamls and jsons there.
        self.acquire_data()  # This is really run the thing and return to the IQ data in self.measurement.
        self.cfg.data.save_measurement(self.data_path, self.measurement, self.attrs)
        self.process_data()
        self.fit_data()
        self.cfg.data.save_measurement(self.data_path, self.measurement, self.attrs)
        self.plot()
        self.n_runs += 1
        self.measurements.append(self.measurement)
        
        
    def check_attribute(self):
        """
        Check the qubits/resonators are always string with 'Q' or 'R'.
        Warn user if any drive_qubits are not being readout without raising error.
        Make sure each qubit has subspace and each resonator has level_to_fit.
        Make sure the prepulse/postpulse is indeed in form of dictionary.
        Check total acquisition point in sequence program.
        Make sure classification is on when heralding is on.
        """
        for qudit in self.qudits:
            assert isinstance(qudit, str), f'The type of {qudit} is not a string!'
            assert qudit.startswith('Q') or qudit.startswith('R'), f'The value of {qudit} is invalid.'
            
        for qubit in self.drive_qubits:
            if f'R{qubit[1]}' not in self.readout_resonators: print(f'Scan: The {qubit} will not be readout!')
        
        assert hasattr(u, self.x_plot_unit), f'The plot unit {self.x_plot_unit} has not been defined.'
        assert self.x_stop >= self.x_start, 'Please use ascending value for x_values.'
        assert len(self.subspace) == len(self.drive_qubits), 'Please specify subspace for each qubit.'
        assert len(self.level_to_fit) == len(self.readout_resonators), 'Please specify subspace for each resonator.'
        assert isinstance(self.prepulse, dict), 'Prepulse must be dictionary like {"Q0":[pulse1, pulse2,...]}'
        assert isinstance(self.postpulse, dict), 'Postpulse must to be dictionary like {"Q0":[pulse1, pulse2,...]}'
        assert self.num_bins <= 131072, 'x_points * n_seqloops cannot exceed 131072! Please use n_pyloops!'
        assert self.classification_enable >= self.heralding_enable, 'Please turn on classification for heralding.'
        

    ##################################################
    def make_sequence(self):
        """
        Generate the self.sequences, which is a dictionary including all sequence \
        dictionaries that we will dump to json file.
        
        Example:
        self.sequences = {'Q3': Q3_sequence_dict,
                          'Q4': Q4_sequence_dict,
                          'R3': R3_sequence_dict,
                          'R4': R4_sequence_dict}
        
        Each sequence dictionary should looks like:
        sequence_dict = {'waveforms': waveforms,
                         'weights': weights,
                         'acquisitions': acquisitions,
                         'program': seq_prog}
        
        Please check the link below for detail:
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/basic_sequencing.html
        """
        self.pulse_df = self.dict_to_DataFrame({}, '', self.qudits)
        
        self.sequences = {qudit:{} for qudit in self.qudits}        
        self.set_waveforms_acquisitions()
        
        self.init_program()
        self.start_loop()
        
        self.add_relaxation()
        if self.cfg.variables['common/heralding']: self.add_heralding()
        self.add_pulse(self.subspace_pulse, 'Subspace')
        self.add_pulse(self.prepulse, 'Prepulse')
        self.add_mainpulse()
        self.add_pulse(self.postpulse, 'Postpulse')
        self.add_readout()
        
        self.end_loop()
    
        
    def set_waveforms_acquisitions(self, **waveform_kwargs):
        """
        Generate waveforms, weights, acquisitions items in self.sequences[qudit].
        
        We skip weights acquisition, and I believe even if we need it,
        we can do it in post process. --Zihao(01/31/2023)
        Please check the link below for detail:
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/binned_acquisition.html
        """
        for qudit in self.qudits:
            if qudit.startswith('Q'):
                length = round(self.cfg['variables.common/qubit_pulse_length'] * 1e9)
                shape = self.cfg.variables[f'{qudit}/pulse_shape']

                waveforms = {'1qMAIN': {'data': get_waveform(length, shape, **waveform_kwargs), 
                                        'index': 0},
                             '1qDRAG': {'data': get_waveform(length, shape+'_derivative', **waveform_kwargs), 
                                        'index': 1}}                
                acquisitions = {}
            
            elif qudit.startswith('R'):
                length = round(self.cfg['variables.common/resonator_pulse_length'] * 1e9)
                shape = self.cfg.variables[f'{qudit}/pulse_shape']

                waveforms = {'RO': {'data': get_waveform(length, shape, **waveform_kwargs), 
                                    'index': 0}}
                
                acquisitions = {'readout':   {'num_bins': self.num_bins, 'index': 0},
                                'heralding': {'num_bins': self.num_bins, 'index': 1}}
            
            
            self.sequences[qudit]['waveforms'] = waveforms
            self.sequences[qudit]['weights'] = {}
            self.sequences[qudit]['acquisitions'] = acquisitions           

        
    def init_program(self):
        """
        Create sequence program and initialize some built-in registers.
        Please do not change the convention here since their function have been hardcoded later.
        """
        for qudit in self.qudits:
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
            self.sequences[qudit]['program'] = program
            
            
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += """
        seq_loop:   
        """
        
     
    def add_xinit(self):
        """
        Set necessary initial value of x parameter to the registers, especially R3 & R4. 
        Child class can super this method to add more initial values.
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += f"""
                    move             {self.x_points},R3    
        """
        
        
    def add_xloop(self):
        """
        Add x_loop to sequence program.
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += """            
        xpt_loop: 
        """
        
        
    def add_sequence_start(self):
        """
        Add sync and phase resetting instruction to sequence program.
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += """     
                #-----------Start-----------
                    wait_sync        8               # Sync at beginning of the loop.
                    reset_ph                         # Reset phase to eliminate effect of previous VZ gate.
                    set_mrk          15              # Enable all markers (binary 1111) for switching on output.
                    upd_param        8               # Update parameters and wait 8ns.
        """
            
        
    def add_relaxation(self):
        """
        Add relaxation to sequence program.
        
        Note from Zihao(03/31/2023):
        Although we can use add_wait to achieve same effect,
        I don't want add a bunch of nonsense 'I' to my pulse_df.  
        """
        relaxation_time_s = self.cfg.variables['common/relaxation_time']
        relaxation_time_us = int( np.ceil(relaxation_time_s*1e6) )
        relaxation = f"""
                #-----------Relaxation-----------
                    move             {relaxation_time_us},R2
        rlx_loop:   wait             1000
                    loop             R2,@rlx_loop
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += relaxation
        
        
    def add_heralding(self, name: str = 'Heralding', add_label: bool = True, 
                      concat_df: bool = True, acq_index: int = 1):
        """
        Add heralding with short delay(relaxation) to the sequence program.
        """
        self.add_readout(name=name, add_label=add_label, concat_df=concat_df, acq_index=acq_index)
        heralding_delay = round(self.cfg.variables['common/heralding_delay'] * 1e9)
        self.add_wait(name=name+'Delay', length=heralding_delay, add_label=add_label, concat_df=concat_df)
        
        
    def add_mainpulse(self):        
        """
        Add main content of the parameter sweep. 
        Suppose to be called by child class.
        """
        print('Scan: The base experiment class has been called. No main pulse will be added.')

    
    def add_readout(self, name='Readout', add_label: bool = True, 
                    concat_df: bool = True, acq_index: int = 0):
        """
        Add readout/heralding with acquisition to sequence program.
        """
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        readout_length_ns = round(self.cfg.variables['common/resonator_pulse_length'] * 1e9)
        self.add_pulse(pulse=self.readout_pulse, lengths=[tof_ns+readout_length_ns],
                       name=name, add_label=add_label, concat_df=concat_df, acq_index=acq_index)
        
        
    def end_loop(self):
        """
        End all loops and add stop to sequence program.
        
        Note from Zihao(03/07/2023):
        In principle, we need to add x_step to the register that store x_value before ending x loop.
        Here I absorb that part into add_mainpulse for continuity and readability.
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += """
                #-----------Stop-----------
                    add              R1,1,R1
                    set_mrk          0               # Disable all markers (binary 0000) for switching off output.
                    upd_param        8               # Update parameters and wait 4ns.     
        """
        

    def end_xloop(self):
        """
        Add end of x loop.
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += """
                    loop             R3,@xpt_loop         
        """
        
        
    def end_seqloop(self):
        """
        End sequence loop and stop the sequence program.
        """
        seq_end = """
                    loop             R0,@seq_loop
                    
                    stop             
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += seq_end


    def add_wait(self, name: str, length: int, add_label: bool = True,
                 concat_df: bool = True, divisor_ns: int = 65528):
        """
        Add wait/Identity to sequence program with user-defined length in [ns].
        The maximum time of 'wait' instruction is 65535, so I divide it into multiple of divisor \
        along with a remainder, and use multiple instruction to achieve it.
        Here I treat it as a pulse, which means we can add label if we want, \
        and we can see it in self.pulse_df
        """
        assert length >= 4, f'The wait time need to be at least 4ns. Now it is {length}.'
        multiple = round(length // divisor_ns)
        remainder = round(length % divisor_ns)
        
        pulse = {qudit: ['I' for i in range(multiple+1)] for qudit in self.qudits}
        lengths = [divisor_ns for i in range(multiple)] + [remainder]
        self.add_pulse(pulse, name, lengths, add_label=add_label, concat_df=concat_df)
    
            
    def add_pulse(self, pulse: dict, name: str, lengths: list = None,
                  add_label: bool = True, concat_df: bool = True, **pulse_kwargs):
        """
        The general method for adding pulses to sequence program.
        We will generate the Pandas DataFrame of prepulse, postpulse, readout, with padded 'I'.
        All qubits and resonators will become the (row) index of dataframe.
        An additional interger attribute 'length' in [ns] will be associated with each column.
        If lengths is shorter than number of pulse, it will be padded using the last length.
        Please remember all labels created in Q1ASM should have different names.
        
        Attributes:
            pulse: {'Q0': ['X180_01'], 'Q1': ['X90_12', 'Y90_12']}.
            lengths: The duration of each pulse(column) in [ns]. Default is single qubit gate time.
            name: String in sequence program to improve readability.
            
        Example of full prepulse DataFrame:
           subspace_0 subspace_1 prepulse_0 prepulse_1
        Q3    X180_01          I     Y90_01          I
        Q4    X180_01    X180_12     Y90_01     Z90_12
        R3          I          I          I          I
        R4          I          I          I          I
        """
        if lengths is None: lengths = [round(self.cfg.variables['common/qubit_pulse_length'] * 1e9)]
        lengths = self.make_it_list(lengths)
        pulse_df = self.dict_to_DataFrame(pulse, name, self.qudits)
        
        for col_name, column in pulse_df.items():
            try:
                column.length = lengths[int(col_name.split('_')[1])]  
                # Use column name as index of lengths list. If list is not long enough, use last index.
            except IndexError:
                column.length = lengths[-1]
                
            for qudit in self.qudits:
                pulse_prog = '' if not add_label else f"""
                # -----------{col_name}-----------
        {col_name}:  """
        
                pulse_prog += pulse_interpreter(cfg = self.cfg, 
                                                qudit = qudit, 
                                                pulse_string = column[qudit], 
                                                length = column.length,
                                                **pulse_kwargs)
                self.sequences[qudit]['program'] += pulse_prog
        
        # Concatenate a larger dataframe while keeping length for user to read/check it.
        if concat_df:
            old_df = self.pulse_df
            self.pulse_df = pd.concat([old_df, pulse_df], axis=1)
            
            for col_name, column in self.pulse_df.items():
                column.length = old_df[col_name].length if col_name in old_df else pulse_df[col_name].length
            
        
        
    ##################################################    
    def save_sequence(self, jsons_path: str = None):
        """
        Create json file of sequence for each sequencer/qudit and save it.
        Allow user to pass a path of directory to save jsons at another place.
        A text file of sequence program will also be saved for reading.
        """
        if jsons_path is None:
            jsons_path = self.jsons_path 

        for qudit, sequence_dict in self.sequences.items():
            file_path = os.path.join(jsons_path, f'{qudit}_sequence.json')
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(sequence_dict, file, indent=4)
                
            txt_file_path = os.path.join(jsons_path, f'{qudit}_sequence_program.txt')
            with open(txt_file_path, 'w', encoding='utf-8') as txt:
                txt.write(sequence_dict['program'])
            
    
    def make_exp_dir(self):
        """
        Create an experiment directory under cfg.data.base_directory.
        Then save a copy of jsons and yamls to experiment directory.
        
        Note from Zihao(02/14/2023):
        If sequence program has problem, then self.cfg.DAC.implement_parameters will raise Error.
        In that case we won't create the junk experiment folder.
        If problem happen after we start_sequencer, then it worth to create the experiment folder.
        Because we may already get some data and want to save it by manually calling save_measurement().
        """
        self.cfg.data.make_exp_dir(experiment_type='_'.join([*self.qudits, self.scan_name]),
                                   experiment_suffix=self.experiment_suffix)
        
        self.data_path = self.cfg.data.data_path
        self.datetime_stamp = self.cfg.data.datetime_stamp
        
        self.cfg.save(yamls_path=self.cfg.data.yamls_path, verbose=False)
        self.save_sequence(jsons_path=self.cfg.data.jsons_path)
        
        for r in self.readout_resonators: os.makedirs(os.path.join(self.data_path, f'{r}_IQplots'))
    
    
    def acquire_data(self):
        """
        Create measurement dictionary, then start sequencer and save data into this dictionary.
        self.measurement should only have resonators' name as keys.
        Inside each resonator should be consistent name of processing or raw data.
        After all loops, the 'Heterodyned_readout' usually has shape (2, n_pyloops, n_seqloops*x_points).
        We will reshape it to (2, n_reps, x_points) later by ProcessManager, where n_reps = n_seqloops * n_pyloops.
        """
        self.measurement = {r: {'raw_readout': [[],[]],  # First element for I, second element for Q.
                                'raw_heralding': [[],[]],
                                'Heterodyned_readout': [[],[]],
                                'Heterodyned_heralding':[[],[]]
                                } for r in self.readout_resonators}
        
        print('Scan: Start sequencer.')
        for i in range(self.n_pyloops):
            self.cfg.DAC.start_sequencer(self.drive_qubits, self.readout_resonators, self.measurement)
            print(f'Scan: Pyloop {i} finished!')
            
            
    def process_data(self):
        """
        Process the data by performing reshape, rotation, average, GMM, fit, plot, etc.
        We keep this layer here to provide possibility to inject functionality between acquire and fit.
        For example, in 2D scan, we need to give it another shape.
        """
        self.cfg.process.process_data(self.measurement, shape=(2, self.n_reps, self.x_points))
        

    def fit_data(self, x: list | np.ndarray = None): 
        """
        Fit data in measurement dictionary and save result back into it.
        We will also keep fit result as attribute self.fit_result for future use.
        The model should be better to pick from qtrlb.processing.fitting.
        data_dict['to_fit'] usually have shape (n_levels, x_points), or (2, x_points) without classification.
        
        Note from Zihao(03/17/2023):
        I leave an interface for x because it's convenient to fit multidimensional scan as 1D scan.
        In that case we can pass whichever axis as horizontal axis here.
        The only cost is to process self.measurement[r]['to_fit'] to correct shape.
        # TODO: make 2D data fitting possible. Now RTS use 1D-like fitting.
        """
        self.fit_result = {r: None for r in self.readout_resonators}
        if self.fitmodel is None: return
        if x is None: x = self.x_values
        
        for i, r in enumerate(self.readout_resonators):
            try:
                level_index = self.level_to_fit[i] - self.cfg[f'variables.{r}/lowest_readout_levels']
                self.fit_result[r] = fit(input_data=self.measurement[r]['to_fit'][level_index],
                                         x=x, fitmodel=self.fitmodel)
                
                params = {v.name:{'value':v.value, 'stderr':v.stderr} for v in self.fit_result[r].params.values()}
                self.measurement[r]['fit_result'] = params
                self.measurement[r]['fit_model'] = str(self.fit_result[r].model)
            except Exception:
                traceback_str = traceback.format_exc()
                print(f'Scan: Failed to fit {r} data. ', traceback_str)
                self.measurement[r]['fit_result'] = None
                self.measurement[r]['fit_model'] = str(self.fitmodel)
                

    def plot(self):
        """
        Plot the experiment result and save them into data directory.
        """
        self.plot_main()
        self.plot_IQ()
        if self.classification_enable: self.plot_populations()


    def plot_main(self, text_loc: str = 'lower right'):
        """
        Plot the main result along with fitting, if not failed.
        Figure will be saved to data directory and show up on python console.
        
        Reference about add_artist method of Axes object:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_artist.html
        """
        for i, r in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{r}/lowest_readout_levels']      
            title = f'{self.datetime_stamp}, {self.scan_name}, {r}'
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            if self.classification_enable:
                ylabel = fr'$P_{{\left|{self.level_to_fit[i]}\right\rangle}}$'
            else:
                ylabel = 'I-Q Coordinate (Rotated) [a.u.]'
            
            fig, ax = plt.subplots(1, 1, dpi=150)
            ax.plot(self.x_values / self.x_unit_value, self.measurement[r]['to_fit'][level_index], 'k.')
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
            
            if self.fit_result[r] is not None: 
                # Raise resolution of fit result for smooth plot.
                x = np.linspace(self.x_start, self.x_stop, self.x_points * 3)  
                y = self.fit_result[r].eval(x=x)
                ax.plot(x / self.x_unit_value, y, 'm-')
                
                # AnchoredText stolen from Ray's code.
                fit_text = '\n'.join([fr'{v.name} = {v.value:0.3g}$\pm${v.stderr:0.1g}' \
                                      for v in self.fit_result[r].params.values()])
                anchored_text = AnchoredText(fit_text, loc=text_loc, prop={'color':'m'})
                ax.add_artist(anchored_text)

            fig.savefig(os.path.join(self.data_path, f'{r}.png'))
            
            
    def plot_IQ(self, 
                IQ_key: str = 'IQrotated_readout', 
                c_key: str = 'GMMpredicted_readout', 
                mask_key: str = 'Mask_heralding'):
        """
        Plot the IQ point for each element in self.x_values.
        Figure will be saved to data directory without show up on python console.
        Additionally, if we enable classification, we use GMM predicted result as colors.
        Furthermore, if we enable heralding, we will also make plot for those data passed heralding test. 
        
        Note from Zihao(03/02/2023):
        The cmap trick is stolen from previous code and assume we have at most 12 levels to plot.
        Only when somebody really try to write this code do they realize how difficult it is.
        With for-for-if three layers nested, this is the best I can do.
        The heralding enable is protected since we have self.check_attribute().
        """
        for r in self.readout_resonators:
            Is, Qs = self.measurement[r][IQ_key]
            left, right = (np.min(Is), np.max(Is))
            bottom, top = (np.min(Qs), np.max(Qs))
            for x in range(self.x_points):
                I = self.measurement[r][IQ_key][0,:,x]
                Q = self.measurement[r][IQ_key][1,:,x]
                c, cmap = (None, None)
                                  
                if self.classification_enable:
                    c = self.measurement[r][c_key][:,x]
                    cmap = LSC.from_list(None, plt.cm.tab10(self.cfg[f'variables.{r}/readout_levels']), 12)

                fig, ax = plt.subplots(1, 1, dpi=150)
                ax.scatter(I, Q, c=c, cmap=cmap, alpha=0.2)
                ax.axvline(color='k', ls='dashed')    
                ax.axhline(color='k', ls='dashed')
                ax.set(xlabel='I', ylabel='Q', title=f'{x}', aspect='equal', 
                       xlim=(left, right), ylim=(bottom, top))
                fig.savefig(os.path.join(self.data_path, f'{r}_IQplots', f'{x}.png'))
                plt.close(fig)
                
                if self.heralding_enable:
                   mask = self.measurement[r][mask_key][:,x] 
                   I_masked = np.ma.MaskedArray(I, mask=mask)
                   Q_masked = np.ma.MaskedArray(Q, mask=mask)
                   c_masked = np.ma.MaskedArray(c, mask=mask)
                   
                   fig, ax = plt.subplots(1, 1, dpi=150)
                   ax.scatter(I_masked, Q_masked, c=c_masked, cmap=cmap, alpha=0.2)
                   ax.axvline(color='k', ls='dashed')    
                   ax.axhline(color='k', ls='dashed')
                   ax.set(xlabel='I', ylabel='Q', title=f'{x}', aspect='equal', 
                          xlim=(left, right), ylim=(bottom, top))
                   fig.savefig(os.path.join(self.data_path, f'{r}_IQplots', f'heralded_{x}.png'))
                   plt.close(fig)


    def plot_populations(self):
        """
        Plot populations for all levels, both with and without readout correction.
        """
        for r in self.readout_resonators:
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            ylabel = 'Probability'
            
            title = f'Uncorrected probability, {self.scan_name}, {r}'
            fig, ax = plt.subplots(1, 1, dpi=150)
            for i, level in enumerate(self.cfg[f'variables.{r}/readout_levels']):
                ax.plot(self.x_values / self.x_unit_value, self.measurement[r]['PopulationNormalized_readout'][i], 
                        c=f'C{level}', ls='-', marker='.', label=fr'$P_{{{level}}}$')
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title, ylim=(-0.05,1.05))
            plt.legend()
            fig.savefig(os.path.join(self.data_path, f'{r}_PopulationUncorrected.png'))
            plt.close(fig)
            
            title = f'Corrected probability, {self.scan_name}, {r}'
            fig, ax = plt.subplots(1, 1, dpi=150)
            for i, level in enumerate(self.cfg[f'variables.{r}/readout_levels']):
                ax.plot(self.x_values / self.x_unit_value, self.measurement[r]['PopulationCorrected_readout'][i], 
                        c=f'C{level}', ls='-', marker='.', label=fr'$P_{{{level}}}$')
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title, ylim=(-0.05,1.05))
            plt.legend()
            fig.savefig(os.path.join(self.data_path, f'{r}_PopulationCorrected.png'))
            plt.close(fig)
        
        
    @staticmethod
    def make_it_list(thing):
        """
        A crucial, life-saving function.
        """
        if isinstance(thing, list):
            return thing
        elif thing == None:
            return []
        else:
            return [thing]


    @staticmethod
    def dict_to_DataFrame(dic: dict, name: str, rows: list, padding: object = 'I'):
        """
        Turn a dictionary into a Pandas DataFrame with padding.
        Each key in dic or element in rows will become index (row) of the DataFrame.
        Each column will be renamed as 'name_0', 'name_1'.
        
        Example:
            dict: {'Q3':['X180_01', 'X180_12'], 'Q4':['Y90_01']}
            name: 'prepulse'
            rows: ['Q3', 'Q4', 'R3', 'R4']
        """
        for row in rows:
            if row not in dic: dic[row] = []
            
        dataframe = pd.DataFrame.from_dict(dic, orient='index')
        dataframe = dataframe.rename(columns={i:f'{name}_{i}' for i in range(dataframe.shape[1])})
        dataframe = dataframe.fillna(padding)        
        return dataframe
    
    
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
        assert freq <= 500e6 and freq >= -500e6, 'The frequency must between +-500MHz.'
        freq_4 = round(freq * 4)
        
        twos_complement_binary_str = format(freq_4 if freq_4 >= 0 else (1 << bit) + freq_4, f'0{bit}b')
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
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
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
                 subspace: str | list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list = None,
                 fitmodel: Model = None):
        
        Scan.__init__(self,
                      cfg=cfg, 
                      drive_qubits=drive_qubits,
                      readout_resonators=readout_resonators,
                      scan_name=scan_name,
                      x_plot_label=x_plot_label, 
                      x_plot_unit=x_plot_unit, 
                      x_start=x_start, 
                      x_stop=x_stop, 
                      x_points=x_points, 
                      subspace=subspace,
                      prepulse=prepulse,
                      postpulse=postpulse,
                      n_seqloops=n_seqloops,
                      level_to_fit=level_to_fit,
                      fitmodel=fitmodel)
        
        self.y_plot_label = y_plot_label
        self.y_plot_unit = y_plot_unit
        self.y_start = y_start
        self.y_stop = y_stop
        self.y_points = y_points
        self.num_bins = self.n_seqloops * self.x_points * self.y_points
        
        assert hasattr(u, self.y_plot_unit), f'The plot unit {self.y_plot_unit} has not been defined.'
        assert self.y_stop >= self.y_start, 'Please use ascending value for y_values.'
        assert self.num_bins <= 131072, \
            'x_points * y_points * n_seqloops cannot exceed 131072! Please use n_pyloops!'
         
        self.y_values = np.linspace(self.y_start, self.y_stop, self.y_points)
        self.y_step = (self.y_stop - self.y_start) / (self.y_points-1) if self.y_points != 1 else 0
        self.y_unit_value = getattr(u, self.y_plot_unit)
            
            
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += f"""            
                    move             {self.y_points},R5    """


    def add_yloop(self):
        """
        Add y_loop to sequence program.
        """
        for qudit in self.qudits:  self.sequences[qudit]['program'] += """            
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += y_end


    def process_data(self):
        """
        Process the data by performing reshape, rotation, average, GMM, fit, plot, etc.
        We keep this layer here to provide possibility to inject functionality between acquire and fit.
        """
        self.cfg.process.process_data(self.measurement, shape=(2, self.n_reps, self.y_points, self.x_points))


    def plot(self):
        # TODO: Finish it.
        self.plot_main()


    def plot_main(self, text_loc: str = 'upper right'):
        """
        Plot population for all levels. 
        If we disable classification, plot both quadrature.
        
        # TODO add fit to the corresponding level, or add connector for adding fit.
        """
        for i, r in enumerate(self.readout_resonators):
            data = self.measurement[r]['to_fit']
            n_subplots = len(data)
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            ylabel = self.y_plot_label + f'[{self.y_plot_unit}]'
            title = f'{self.datetime_stamp}, {self.scan_name}, {r}'
            
            fig, ax = plt.subplots(1, n_subplots, figsize=(7 * n_subplots, 8), dpi=150)

            for i in range(n_subplots):
                level = self.cfg[f'variables.{r}/readout_levels'][i]
                this_title = title + fr'P_{{{level}}}' if self.classification_enable else title
                
                image = ax[i].imshow(data[i], cmap='RdBu_r', interpolation='none', aspect='auto', 
                                     origin='lower', extent=[np.min(self.x_values) / self.x_unit_value, 
                                                             np.max(self.x_values) / self.x_unit_value, 
                                                             np.min(self.y_values) / self.y_unit_value, 
                                                             np.max(self.y_values) / self.y_unit_value])
                ax[i].set(title=this_title, xlabel=xlabel, ylabel=ylabel)
                fig.colorbar(image, ax=ax[i], label='Probability/Coordinate', location='top')
                
            fig.savefig(os.path.join(self.data_path, f'{r}.png'))











