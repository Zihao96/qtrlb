import numpy as np
import pandas as pd
from qtrlb.utils.waveforms import get_waveform  # TODO: Write it.
from qtrlb.utils.pulse import pulse_interpreter  # TODO: Write it. Make sure Empty DataFrame works.




class Scan:
    """ Base class for all parameter-sweep experiment.
        The framework of how experiment flow will be constructed here.
        It should be used as parent class of specific scan rather than be instantiated directly.
        
        Attributes:
            config: A MetaManager.
            drive_qubits: 'Q2', or ['Q3', 'Q4']. User has to specify the subspace.
            readout_resonators: 'R3' or ['R1', 'R5'].
            subspace: '12' or ['01', '01'], length should be save as drive_qubits.
            prepulse: {'Q0': ['Q0/X180_01'], 'Q1': ['Q0/X90_12', 'Q1/Y90_12']}
            postpulse: Same requirement as prepulse.
    """
    def __init__(self, 
                 config, 
                 scan_name: str,
                 x_label: str, 
                 x_unit, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 scan_start: float | list, 
                 scan_stop: float | list, 
                 npoints: int, 
                 heralding: bool = False,
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 fitmodel = None):
        self.config = config
        self.scan_name = scan_name
        self.x_label = x_label
        self.x_unit = x_unit
        self.drive_qubits = self.make_it_list(drive_qubits)
        self.readout_resonators = self.make_it_list(readout_resonators)
        self.scan_start = self.make_it_list(scan_start)
        self.scan_stop = self.make_it_list(scan_stop)
        self.npoints = npoints
        self.heralding = heralding
        self.subspace = subspace if subspace is not None else ['01']*len(drive_qubits)
        self.prepulse = prepulse if prepulse is not None else {}
        self.postpulse = postpulse if postpulse is not None else {}
        self.fitmodel = fitmodel
        
        self.qudits = self.drive_qubits + self.readout_resonators
        self.heralding_enable = self.config.varman['commmon/heralding']
        self.initialize()
        
        
    def run(self, 
            experiment_suffix: str = '',
            n_reps: int  = 1000):
        self.experiment_suffix = experiment_suffix
        self.n_reps = n_reps
        
        self.make_sequence() 
        self.upload_sequence()
        self.acquire_data()  # This is really run the thing and return to the IQ data.
        self.analyze_data()
        self.plot()
        
        
    def initialize(self):
        """
        Check and reshape the input parameters.
        Configure the Qblox based on drive_qubits and readout_resonators using in this scan.
        Generate self.full_prepulse_df.
        Add subspace prepulse and pad the prepulse/postpulse with 'I'.
        We call implement_parameters methods here instead of during init/load of DACManager,
        because we want those modules/sequencers not being used to keep their default status.
        """     
        self.check_attribute()
        
        # TODO: Gain and NCO freq are unnecessary since we will change them in Q1ASM program anyway.
        self.config.DAC.implement_parameters(qubits=self.drive_qubits, 
                                             resonators=self.readout_resonators,
                                             subspace=self.subspace)
        
        self.generate_prepulse_postpulse_dataframe()        
        
        
    def check_attribute(self):
        """
        Check the qubits/resonators are always string with 'Q' or 'R'.
        Warn user if any drive_qubits are not being readout without raising error.
        Make sure each qubit has a scan_start, scan_stop, subspace.
        Make sure the prepulse/postpulse is indeed in form of dictionary.
        """
        for qudit in self.qudits:
            assert isinstance(qudit, str), f'The type of {qudit} is not a string!'
            assert qudit.startswith('Q') or qudit.startswith('R'), f'The value of {qudit} is invalid.'
            
        for qubit in self.drive_qubits:
            if f'R{qubit[1:]}' not in self.readout_qubits: print(f'The {qubit} will not be readout!')
        
        assert len(self.scan_start) == len(self.drive_qubits), 'Please specify scan_start for each qubit.'
        assert len(self.scan_stop) == len(self.drive_qubits), 'Please specify scan_stop for each qubit.'
        assert len(self.subspace) == len(self.drive_qubits), 'Please specify subspace for each qubit.'
        assert isinstance(self.prepulse, dict), 'Prepulse must be dictionary like {"Q0":[pulse1, pulse2,...]}'
        assert isinstance(self.postpulse, dict), 'Postpulse must to be dictionary like {"Q0":[pulse1, pulse2,...]}'


    def generate_prepulse_postpulse_dataframe(self):
        """
        Generate the Pandas DataFrame of prepulse, postpulse, with padded 'I'.
        Both subspace and input prepulse will be included into prepulse.
        All qubits and resonators will become the (row) index of dataframe.
        """

        for qudit in self.qudits:
            if qudit not in self.prepulse: self.prepulse[qudit] = []
            if qudit not in self.postpulse: self.postpulse[qudit] = []

        self.subspace_pulse = {}
        for q, ss in zip(self.drive_qubits, self.subspace):
            self.subspace_pulse[q] = [f'{q}/X180_{l}{l+1}' for l in range(int(ss[0]))]
        for r in self.readout_resonators:
            self.subspace_pulse[r] = []


        # Generate the DataFrame
        subspace_df = pd.DataFrame.from_dict(self.subspace_pulse, orient='index')
        self.subspace_df = subspace_df.rename(columns={i:f'subspace_{i}' for i in range(subspace_df.shape[1])})

        prepulse_df = pd.DataFrame.from_dict(self.prepulse, orient='index')
        self.prepulse_df = prepulse_df.rename(columns={i:f'prepulse_{i}' for i in range(prepulse_df.shape[1])})

        full_prepulse_df = pd.concat([subspace_df, prepulse_df], axis=1)
        self.full_prepulse_df = full_prepulse_df.fillna('I')

        postpulse_df = pd.DataFrame.from_dict(self.postpulse, orient='index')
        postpulse_df = postpulse_df.rename(columns={i:f'postpulse_{i}' for i in range(postpulse_df.shape[1])})
        self.postpulse_df = postpulse_df.fillna('I')


    def make_sequence(self):
        """
        Generate the self.sequences, which is a dictionary including all sequence dictionaries
        we will dump to json file.
        
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
        self.sequences = {qudit:{} for qudit in self.qudits}        
        
        self.set_waveforms_acquisitions()
        self.init_program()
    
        
    def set_waveforms_acquisitions(self):
        """
        Generate waveforms, weights, acquisitions items in self.sequences[qudit].
        
        We skip weights acquisition, and I believe even if we need it,
        we can do it in post process. --Zihao(01/31/2023)
        Please check the link below for detail:
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/binned_acquisition.html
        """
        for qudit in self.qudits:
            pulse_type = 'qubit' if qudit.startswith('Q') else 'resonator'
            waveforms = {qudit: {'data': get_waveform(self.config.varman[f'common/{pulse_type}_pulse_length'], 
                                                      self.config.varman[f'{qudit}/pulse_shape']), 
                                 'index': 0}}
            acquisitions = {'readout':   {'num_bins': self.npoints, 'index': 0},
                            'heralding': {'num_bins': self.npoints, 'index': 1}}
            
            self.sequences[qudit]['waveforms'] = waveforms
            self.sequences[qudit]['weights'] = {}
            self.sequences[qudit]['acquisitions'] = acquisitions           

        
    def init_program(self):
        
        for qudit in self.qudits:
            program = """
            # R0 is the value of main parameter of 1D Scan.
            # R1 is the count of repetition for algorithm or npoints for parameter sweep.
            # R2 is the relaxation time in microseconds.
            # Other register for backup.
            
                        wait_sync        4
                        move             0,R0
                        move             0,R1
                        move             0,R2
                        move             0,R3
                        move             0,R4
                        move             0,R5
            
            main_loop:  wait_sync        4                               # Sync at beginning of the loop.
                        reset_ph                                         # Reset phase to eliminate effect of previous VZ gate.
                        set_mrk          15                              # Enable all markers (binary 1111) for switching on output.
                        upd_param        4                               # Update parameters and wait 4ns.
            """
            self.sequences[qudit]['program'] = program
        
        
    def add_relaxation(self):
        
        relaxation_time_s = self.config.varman['commmon/relaxation_time']
        relaxation_time_us = int( np.ceil(relaxation_time_s*1e6) )
        relaxation = f"""
        #-----------Relaxation-----------
                    move             {relaxation_time_us},R2
        relx_loop:  wait             1000
                    loop             R2,@relx_loop
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += relaxation
        
        
    # TODO: write it.
    def add_heralding(self):
        return 
    
    
    def add_prepulse(self):
        
        for qudit in self.qudits:
            prepulse = """
            #-----------Prepulse-----------
            """
            init_pulse_str = self.full_prepulse_df.loc[qudit, :]  # Pandas Series, each element is like 'Q3/X180_01'
            prepulse += pulse_interpreter(init_pulse_str)  # Long String, part of the sequence program.
            
            self.sequences[qudit]['program'] += prepulse
        

    def add_mainpulse(self):        
        """
        Suppose to be called by child class.
        """
        print('The base experiment class has been called. No main pulse will be added.')
        return

        
    def add_postpulse(self):
        
        for qudit in self.qudits:
            postpulse = """
            #-----------postpulse-----------
            """
            init_pulse_str = self.postpulse_df.loc[qudit, :]
            postpulse += pulse_interpreter(init_pulse_str)
            
            self.sequences[qudit]['program'] += postpulse


    # TODO: Maybe make RO as a pulse in pulse_interpreter.
    def add_readout(self):
        
        readout_length_s = self.config.varman['commmon/resonator_pulse_length']
        readout_length_ns = int( np.ceil(readout_length_s*1e9) )
        tof_s = self.config.varman['commmon/tof']
        tof_ns = int( np.ceil(tof_s*1e9) )
        
        for qubit in self.drive_qubits:
            readout = f"""
            #-----------readout-----------
                        wait             {tof_ns + readout_length_ns} 
            """
            self.sequences[qubit]['program'] += readout
            
        for resonator in self.readout_resonators:
            readout = f"""
            #-----------readout-----------
                        play             0,0,{tof_ns} 
                        acquire          0,R1,{readout_length_ns}
                        
            """
            self.sequences[resonator]['program'] += readout 
            

    def add_stop(self):
        
        stop = f"""
        #-----------stop-----------
                    add              R1,1,R1
                    set_mrk          0                               # Disable all markers (binary 0000) for switching off output.
                    upd_param        4                               # Update parameters and wait 4ns.
                    jlt              R1,{self.npoints},@main_loop
                    
                    stop             
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += stop


    # TODO: Set a maximum of it.
    def add_wait(self, time: int):
        
        wait = f"""
        #-----------wait-----------
                    wait             {time}                               
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += wait
    
            
    # TODO: write it.        
    def add_pulse(self, pulse: dict):
        pass
        
        
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














































