from qtrl.utils import Config
import warnings

class VariableManager(Config):
    """Thin wrapper over a config_file to help with modulation management.
    """
    def load(self, config_raw=None):
        super().load(config_raw=config_raw)

        try:
            qubit_lo = self.get('hardware/qubit_LO')
            readout_lo = self.get('hardware/readout_LO')
        except KeyError:
            warnings.warn('Could not find the LO frequencies for qubit drive and readout')
            return
        
        # Add 'mod_freq' and 'anharmonicity' for each subspace. Need to generalize 'anharmonicity' to more subspace.
        for key in self.keys():
            if key.startswith('Q') and key[1:].isdecimal():
                self.set_dict(f'{key}/anharmonicity', self.get(f'{key}/12/freq') - self.get(f'{key}/01/freq'))
                # Create modulation(AWG) frequency for each subspace
                for subspace in self.get(key): #For the new structure 'Q1/01/freq', 'Q1/12/freq', etc... -RP (9/2/21)
                    try:
                        freq = self.get(f'{key}/{subspace}/freq')
                        self.set_dict(f'{key}/{subspace}/mod_freq', freq - qubit_lo)
                    except KeyError:
                        print(f'Cannot get frequency for qubit {key} subspace {subspace}.')
                        pass       
                
                # Readouts. Add 'res_mod_freq' and 'n_levels' for each qudit.
                if 'res_freq' in self.get(key):
                    freq = self.get(f'{key}/res_freq')
                    self.set_dict(f'{key}/res_mod_freq', freq - readout_lo)
                    
                if 'IQ_means' in self.get(key):
                    n_levels = len(self.get(f'{key}/IQ_means'))  # IQ_means is n*2 matrix/list.
                    self.set_dict(f'{key}/n_levels', n_levels)


    def set_vars(self, new_settings):
        """ Given a list of dictionaries new_settings, set variables"""
        if not isinstance(new_settings, list):
            new_settings = [new_settings]
        for new_setting_dict in new_settings:  # A dict in a list of dicts.
            for key in new_setting_dict:
                print(key, new_setting_dict[key])
                self.set(key, new_setting_dict[key])
        self.load()
        
