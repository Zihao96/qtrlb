from qtrlb.config.config import Config
import numpy as np


class VariableManager(Config):
    """ This is a thin wrapper over the Config class to help with modulation management.
        The load() method will be applied in its __init__.
    
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
        including mod_freq for AWG, anharmonicity, n_readout_levels.
        Additionally, it will check the shape of those IQ matrices inside 
        Variables.yaml. If their shapes are not compatible with readout_levels,
        default compatible matrices will be generated without saving.
        """
        super().load()

        # Loop all sub-dictionary with key 'Q0', 'Q1', 'Q2' etc.
        for key in self.keys():
            if key.startswith('Q'):
                
                # Set AWG frequency for RO.
                self.set(f'{key}/res_mod_freq', self[f'{key}/res_freq']-self['readout/readout_LO'], which='dict')
                
                # Make sure readout_levels are in ascending order.
                self.set(f'{key}/readout_levels', sorted(self[f'{key}/readout_levels']), which='dict')
                
                self.set(f'{key}/lowest_readout_levels', self[f'{key}/readout_levels'][0], which='dict')
                self.set(f'{key}/highest_readout_levels', self[f'{key}/readout_levels'][-1], which='dict')
                self.set(f'{key}/n_readout_levels', len(self[f'{key}/readout_levels']), which='dict')
                
                # Check the shape of IQ-matrices.
                try:
                    assert (self[f'{key}/n_readout_levels'] == len(self[f'{key}/IQ_means']) 
                            == len(self[f'{key}/IQ_covariances']) == len(self[f'{key}/corr_matrix']))
                except AssertionError:
                    print(f'The shapes of IQ matrices in {key} are not compatible with its readout_levels.'
                          +'New matrices will be generated. Please save it by calling cfg.save()')
                    self.generate_default_IQ_matrices(qubit=key, n_readout_levels=self[f'{key}/n_readout_levels'])
                    
                # Loop all subspace.
                for subspace in self[f'{key}']:
                    if not subspace.isdecimal(): continue
                
                    # Set AWG frequency for each subspace.
                    self.set(f'{key}/{subspace}/mod_freq', 
                             self[f'{key}/{subspace}/freq']-self[f'{key}/readout_LO'], 
                             which='dict')
                    
                    # Set anharmonicity for each subspace.
                    if subspace == '01': continue
                    self.set(f'{key}/{subspace}/anharmonicity', 
                             self[f'{key}/{subspace}/freq']-self[f'{key}/{int(subspace)-11}/freq'], 
                             which='dict')  # Yep, I made it. --Zihao(01/25/2023)     
  
                        
    def generate_default_IQ_matrices(self, qubit: str, n_readout_levels: int, save_raw: bool = False):
        """
        Set default IQ matrices to config_raw and config_dict based on number of levels to readout.
        Leave the knob of saving config_raw here in case we want to change mind.
        Make it separated so that people can easily change it depends on the exact scale/value of IQ coordinates.
        """
        self.set(f'{qubit}/corr_matrix', np.identity(n_readout_levels).tolist(), save_raw=save_raw)
        self.set(f'{qubit}/IQ_covariances', [1 for i in range(n_readout_levels)], save_raw=save_raw)
        self.set(f'{qubit}/IQ_means', [[i,i] for i in range(n_readout_levels)], save_raw=save_raw)




