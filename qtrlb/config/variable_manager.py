import qtrlb.utils.units as u
from qtrlb.config.config import Config


class VariableManager(Config):
    """ This is a thin wrapper over the Config class to help with variables management.
        The load() method will be called once in its __init__.
    
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
            variable_suffix: 'EJEC' or 'ALGO'. A underscroll will be added in this layer.
    """
    def __init__(self, 
                 yamls_path: str, 
                 variable_suffix: str = ''):
        super().__init__(yamls_path=yamls_path, 
                         suffix='variables',
                         variable_suffix='_'+variable_suffix)
        self.load()
        
    
    def load(self):
        """
        Run the parent load, then generate a seires of items in config_dict
        including tones, mod_freq for AWG, anharmonicity, n_readout_levels.
        If asked, it will check the value inside the dictionary to make
        sure they are valid.
        """
        super().load()
        
        qubits_list = [key for key in self.keys() if key.startswith('Q')]
        resonators_list = [key for key in self.keys() if key.startswith('R')]
        self.set('qubits', qubits_list, which='dict')  # Keep the new key start with lowercase!
        self.set('resonators', resonators_list, which='dict')
        
        self.set_parameters()
        if self['common/check_module_LO_sequencer']: self.check_module_sequencer_LO()
   
    
    def set_parameters(self):
        """
        Set mod_freq for AWG, anharmonicity, n_readout_levels, etc, into config_dict.
        Most importantly, it will generate the key 'tones' whose value looks like: 
        ['Q0/01', 'Q0/12', 'Q1/01', 'Q1/12', 'Q2/01', 'Q2/12', 'R0', 'R1', 'R2'].
        """
        tones_list = []

        for q in self['qubits']:
            for subspace in self[f'{q}']:
                if not subspace.isdecimal(): continue
                tones_list.append(f'{q}/{subspace}')

                # Check the DRAG_weight is in range.
                assert -1 <= self[f'{q}/{subspace}/amp_180'] * self[f'{q}/{subspace}/DRAG_weight'] < 1, \
                    f'DRAG weight of {q}/{subspace} is out of range.'
            
                # Set AWG frequency for each subspace.
                self.set(f'{q}/{subspace}/mod_freq', 
                         self[f'{q}/{subspace}/freq']-self[f'{q}/qubit_LO'], 
                         which='dict')
                
                # Set anharmonicity for each subspace.
                if subspace == '01': continue
                self.set(f'{q}/{subspace}/anharmonicity', 
                         self[f'{q}/{subspace}/freq']-self[f'{q}/{(int(subspace)-11):02}/freq'], 
                         which='dict')  # Yep, I made it. --Zihao(01/25/2023)  
                
        for r in self['resonators']:
            tones_list.append(r)

            # Set AWG frequency for Resonator.
            self.set(f'{r}/mod_freq', self[f'{r}/freq']-self[f'{r}/resonator_LO'], which='dict')
            
            # Make sure readout_levels are in ascending order.
            self.set(f'{r}/readout_levels', sorted(self[f'{r}/readout_levels']), which='dict')
            
            self.set(f'{r}/lowest_readout_levels', self[f'{r}/readout_levels'][0], which='dict')
            self.set(f'{r}/highest_readout_levels', self[f'{r}/readout_levels'][-1], which='dict')
            self.set(f'{r}/n_readout_levels', len(self[f'{r}/readout_levels']), which='dict')
        
        self.set('tones', tones_list, which='dict')


    def check_module_sequencer_LO(self):
        """
        For qubits/resonators using same module, check they are using different sequencer.
        Furthermore, if they use same output, check they are using same LO frequency.

        Note from Zihao(04/10/2023):
        I choose not to check the sequencer conflict here.
        It's because of the possible demand for flexibility of sequencer mapping.
        """
        for i, qubit in enumerate(self['qubits']):
            for qubjt in self['qubits'][i+1:]:
                if (self[f'{qubjt}/module'] == self[f'{qubit}/module']
                    and 
                    self[f'{qubjt}/out'] == self[f'{qubit}/out']):
                        assert self[f'{qubjt}/qubit_LO'] == self[f'{qubit}/qubit_LO'], \
                        f'{qubit} and {qubjt} are using same output port with different LO frequency!'

        for i, resonator in enumerate(self['resonators']):    
            for resonatos in self['resonators'][i+1:]:
                if self[f'{resonatos}/module'] == self[f'{resonator}/module']:
                    
                    assert self[f'{resonatos}/sequencer'] != self[f'{resonator}/sequencer'], \
                    f'{resonator} and {resonatos} are using same sequencer!'
                    
                    assert self[f'{resonatos}/resonator_LO'] == self[f'{resonator}/resonator_LO'], \
                    f'{resonatos} and {resonator} are using same output port with different LO frequency!'        


    def transmon_parameters(self, transmon: str, chi_kHz: float = None):
        """
        All return values are in [GHz].
        
        Calculate transmon property like EJ, EC, g, etc from values in variables.yaml.
        Notice the full dispersive shift/ac Stark shift is 2 * chi.
        Notice fr is a little bit tricky, but doesn't influence the result too much.
        """
        try:
            from qtrlb.utils.transmon_parameters3 import falpha_to_EJEC, get_bare_frequency
        except ModuleNotFoundError:
            print('Missing the module to run such function')
            return

        assert transmon.startswith('Q'), 'Transmon has to be string like "Q3", "Q0".'
        resonator = f'R{transmon[1:]}'
        f01_GHz = self[f'{transmon}/01/freq'] / u.GHz
        alpha_GHz = self[f'{transmon}/12/anharmonicity'] / u.GHz
        fr_GHz = self[f'{resonator}/freq'] / u.GHz
        
        # The return values depends on whether user has run ReadoutFrequencyScan.
        if chi_kHz is None:
            EJ, EC = falpha_to_EJEC(f01_GHz, alpha_GHz)
            return EJ, EC
        else:
            chi_GHz = chi_kHz * u.kHz / u.GHz
            f01_b, alpha_b, fr_b, g01 = get_bare_frequency(f01_GHz, alpha_GHz, fr_GHz, chi_GHz)
            EJ, EC = falpha_to_EJEC(f01_b, alpha_b)
            return EJ, EC, g01





















