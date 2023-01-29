from qtrlb.config.config import Config
import numpy as np


class VariableManager(Config):
    """ This is a thin wrapper over the Config class to help with variables management.
        The load() method will be called once in its __init__.
    
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
            variable_suffix: 'EJEC' or 'ALGO'. A underscroll will be added in this layer.
    """
    def __init__(self, yamls_path: str, variable_suffix: str = ''):
        super().__init__(yamls_path=yamls_path, 
                         suffix='Variables',
                         variable_suffix='_'+variable_suffix)
        self.load()
    
    
    def load(self):
        """
        Run the parent load, then generate a seires of items in config_dict
        including qubits, mod_freq for AWG, anharmonicity, n_readout_levels.
        Additionally, it will check the value inside the dictionary to make
        sure they are valid.
        """
        super().load()
        
        qubits_list = [key for key in self.keys() if key.startswith('Q')]
        self.set('qubits', qubits_list, which='dict')
        
        self.set_parameters()
        self.check_IQ_matrices()
        self.check_qubits_module_and_LO()
   
    
    def set_parameters(self):
        """
        Set mod_freq for AWG, anharmonicity, n_readout_levels, etc, into config_dict.
        """
        for qubit in self['qubits']:
            # Set AWG frequency for RO.
            self.set(f'{qubit}/res_mod_freq', self[f'{qubit}/res_freq']-self['readout/readout_LO'], which='dict')
            
            # Make sure readout_levels are in ascending order.
            self.set(f'{qubit}/readout_levels', sorted(self[f'{qubit}/readout_levels']), which='dict')
            
            self.set(f'{qubit}/lowest_readout_levels', self[f'{qubit}/readout_levels'][0], which='dict')
            self.set(f'{qubit}/highest_readout_levels', self[f'{qubit}/readout_levels'][-1], which='dict')
            self.set(f'{qubit}/n_readout_levels', len(self[f'{qubit}/readout_levels']), which='dict')
                
            # Loop all subspace.
            for subspace in self[f'{qubit}']:
                if not subspace.isdecimal(): continue
            
                # Set AWG frequency for each subspace.
                self.set(f'{qubit}/{subspace}/mod_freq', 
                         self[f'{qubit}/{subspace}/freq']-self[f'{qubit}/qubit_LO'], 
                         which='dict')
                
                # Set anharmonicity for each subspace.
                if subspace == '01': continue
                self.set(f'{qubit}/{subspace}/anharmonicity', 
                         self[f'{qubit}/{subspace}/freq']-self[f'{qubit}/{(int(subspace)-11):02}/freq'], 
                         which='dict')  # Yep, I made it. --Zihao(01/25/2023)  
        
        
    def check_IQ_matrices(self):
        """
        Check the shape of those IQ matrices inside Variables.yaml. 
        If their shapes are not compatible with readout_levels,
        default compatible matrices will be generated without saving.
        """
        for qubit in self['qubits']:
            try:
                assert (self[f'{qubit}/n_readout_levels'] == len(self[f'{qubit}/IQ_means']) 
                        == len(self[f'{qubit}/IQ_covariances']) == len(self[f'{qubit}/corr_matrix']))
            except AssertionError:
                print(f'The shapes of IQ matrices in {qubit} are not compatible with its readout_levels.'
                      +'New matrices will be generated. Please save it by calling cfg.save()')
                
                self[f'{qubit}/corr_matrix'] = np.identity(self[f'{qubit}/n_readout_levels']).tolist()
                self[f'{qubit}/IQ_covariances'] = [1 for i in range(self[f'{qubit}/n_readout_levels'])]
                self[f'{qubit}/IQ_means'] = [[i,i] for i in range(self[f'{qubit}/n_readout_levels'])]


    def check_qubits_module_and_LO(self):
        """
        Check that no more than 6 qubits using same module, and for qubit using same module and output, 
        check they are using same LO frequency.
        It's because each module on Qblox has only 6 sequencer and one LO for each output port.
        """
        for i, qubit in enumerate(self['qubits']):
            n_sequencer_needed = 1
            qubit_module = self[f'{qubit}/module']
            qubit_out = self[f'{qubit}/out']
            qubit_lo_freq = self[f'{qubit}/qubit_LO']
            
            for qubjt in self['qubits'][i+1:]:
                qubjt_module = self[f'{qubjt}/module']
                qubjt_out = self[f'{qubjt}/out']
                qubjt_lo_freq = self[f'{qubjt}/qubit_LO']
                
                if qubjt_module == qubit_module:
                    n_sequencer_needed += 1
                    if qubjt_out == qubit_out:
                        assert qubjt_lo_freq==qubit_lo_freq, \
                        f'{qubit} and {qubjt} are using same output port with different LO frequency!'
                
            assert n_sequencer_needed <= 6, 'More than 6 qubits are using same Qblox Module!'



























