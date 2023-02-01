import numpy as np
from qtrlb.utils.waveforms import get_waveform  # TODO: Write it.





class Scan:
    """ Base class for all parameter-sweep experiment.
        The framework of how experiment flow will be constructed here.
        It should be used as parent class of specific scan rather than be instantiated directly.
        
        Attributes:
            config: A MetaManager.
            drive_qubits: 'Q2', or ['Q3', 'Q4'].
            readout_resonators: 'R3' or ['R1', 'R5'].
            subspace: '12' or ['01', '01'], length should be save as drive_qubits.
            prepulse: 'Q4/X180_01' or [['Q3/Y90_01', 'Q3/X180_12'], ['']],
                       length should be save as drive_qubits.
            postpulse: Same requirement as prepulse.
    """
    def __init__(self, 
                 config, 
                 scan_name: str,
                 x_label: str, 
                 x_unit, 
                 scan_start: float | list, 
                 scan_stop: float | list, 
                 npoints: int, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 subspace: str | list = ['01'],
                 prepulse: str | list = None,
                 postpulse: str | list = None,
                 fitmodel = None):
        self.config = config
        self.scan_name = scan_name
        self.x_label = x_label
        self.x_unit = x_unit
        self.scan_start = scan_start
        self.scan_stop = scan_stop
        self.npoints = npoints
        self.drive_qubits = self.make_it_list(drive_qubits)
        self.readout_resonators = self.make_it_list(readout_resonators)
        self.subspace = self.make_it_list(subspace)
        self.prepulse = self.make_it_list(prepulse)
        self.postpulse = self.make_it_list(prepulse)
        self.fitmodel = fitmodel
        
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
        We call implement_parameters methods here instead of during init/load of DACManager,
        because we want those modules/sequencers not being used to keep their default status.
        """     
        # TODO: Need to make sure prepulse has correct 2D shape.
        self.check_attribute()
        
        # TODO: Some of the parameters are unnecessary since we will change them in Q1ASM program anyway.
        self.config.DAC.implement_parameters(qubits=self.drive_qubits, 
                                             resonators=self.readout_resonators,
                                             subspace=self.subspace)
        
        
    def check_attribute(self):
        """
        Check the qubits/resonators are always string with 'Q' or 'R'.
        Warn user if any drive_qubits are not being readout without raising error.
        Make sure each qubit has a subspace.
        Make sure each qubit has a prepulse/postpulse list.
        """
        for qudit in (self.drive_qubits + self.readout_resonators):
            assert isinstance(qudit, str), f'The type of {qudit} is not a string!'
            assert qudit.startswith('Q') or qudit.startswith('R'), f'The value of {qudit} is invalid.'
        
        for qubit in self.drive_qubits:
            if f'R{qubit[1:]}' not in self.readout_qubits:
                print(f'The {qubit} will not be readout!')
        
        assert len(self.subspace) == len(self.drive_qubits), 'Please specify subspace for each qubit.'
        assert len(self.prepulse) == len(self.drive_qubits), 'Please specify prepulse for each qubit.'
        assert len(self.postpulse) == len(self.drive_qubits), 'Please specify postpulse for each qubit.'
        
        
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
        self.sequences = {}        
        
        for i, qudit in enumerate(self.drive_qubits + self.readout_resonators):
            self.sequences[qudit] = {}
            
            
            self.set_waveforms(qudit=qudit)
            self.set_weights(qudit=qudit)
            self.set_acquisitions(qudit=qudit)
            self.set_program(qudit=qudit)
        
        
    def set_waveforms(self, qudit: str):
        """
        Generate waveforms items in self.sequences[qudit].
        The input can be either 'Q#' or 'R#'.
        """
        waveforms = {qudit: {'data': get_waveform(self.config.varman[f'{qudit}/pulse_length'], 
                                                  self.config.varman[f'{qudit}/pulse_shape']), 
                             'index': 0}}
        
        self.sequences[qudit]['waveforms'] = waveforms
        
        
    def set_weights(self, qudit: str):
        """
        Generate weights items in self.sequences[qudit].
        The input can be either 'Q#' or 'R#'.
        
        It's about weights acquisition, and I believe even if we need it,
        we can do it in post process. --Zihao(01/31/2023)
        Please check the link below for detail:
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/binned_acquisition.html
        """
        self.sequences[qudit]['weights'] = {}       


    def set_acquisitions(self, qudit: str):
        """
        Generate acquisitions items in self.sequences[qudit].
        The input can be either 'Q#' or 'R#'.
        """
        acquisitions = {'readout':   {'num_bins': self.npoints,
                                      'index': 0},
                        'heralding': {'num_bins': self.npoints,
                                      'index': 1}}
        
        self.sequences[qudit]['acquisitions'] = acquisitions
        
        
    # TODO: Finish it.
    def set_program(self, qudit: str):
        """
        Generate Q1ASM program items in self.sequences[qudit].
        The input can be either 'Q#' or 'R#'.
        """
        
        program = self.init_program()
        program = self.add_relaxation(program)
        program = self.add_heralding(qudit, program)
        program = self.add_subspace_prepulse(qudit, program)
        program = self.add_prepulse(qudit, program)
        program = self.add_mainpulse(qudit, program)
        program = self.add_postpulse(qudit, program)
        program = self.add_readout(qudit, program)
        program = self.add_stop(program)
        
        self.sequences[qudit]['program'] = program
        
        
    def init_program(self):
        program = """
        # R0 is the main parameter of 1D Scan.
        # R1 is the count of repetition.
        # R2 is the relaxation time in microseconds.
        
                    wait_sync        4
                    move             0,R0
                    move             0,R1
                    move             0,R2
        
        main_loop:  wait_sync        4                               # Sync at beginning of the loop  
        """
        return program
        
        
    def add_relaxation(self, program: str):
        relaxation_time_s = self.config.varman['commmon/relaxation_time']
        relaxation_time_us = int( np.ceil(relaxation_time_s*1e6) )
        relaxation = f"""
                    move             {relaxation_time_us},R2
        relx_loop:  wait             1000
                    loop             R2,@relx_loop
        """
        return program + relaxation
        
        
    # TODO: write it.
    def add_heralding(self, qudit: str, program: str):
        return program
    
    
    # TODO: Just take a try, don't be afraid.
    def add_subspace_prepulse(self, qudit: str, program: str):
        qubit_ss_prepulse = """
        
        """

        resonator_ss_prepulse = """
        
        """       
            
        subspace_prepulse = qubit_ss_prepulse if qudit.startswith('Q') else resonator_ss_prepulse
        
        return program + subspace_prepulse
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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

















































