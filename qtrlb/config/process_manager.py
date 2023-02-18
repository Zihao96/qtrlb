import numpy as np
from lmfit import Model
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager
from qtrlb.processing.processing import rotate_IQ, gmm_predict, normalize_population, fit


class ProcessManager(Config):
    """ This is a thin wrapper over the Config class to help with measurement process management.
        The load() method will be called once in its __init__.
    
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
            variable_suffix: 'EJEC' or 'ALGO'. A underscroll will be added in this layer.
    """
    def __init__(self, 
                 yamls_path: str, 
                 varman: VariableManager):
        super().__init__(yamls_path=yamls_path, 
                         suffix='process',
                         varman=varman)
        self.load()
        
        
    def load(self):
        """
        Run the parent load, then check the shape of IQ matrices.
        """
        super().load()
        
        resonators_list = [key for key in self.keys() if key.startswith('R')]
        self.set('resonators', resonators_list, which='dict')  # Keep the new key start with lowercase!
        self.check_IQ_matrices()
        
        
    def check_IQ_matrices(self):
        """
        Check the shape of those IQ matrices inside process.yaml. 
        If their shapes are not compatible with readout_levels,
        default compatible matrices will be generated without saving.
        """
        for r in self['resonators']:
            self.set(f'{r}/n_readout_levels', len(self[f'{r}/readout_levels']), which='dict')
            
            try:
                assert (self[f'{r}/n_readout_levels'] == len(self[f'{r}/IQ_means']) 
                        == len(self[f'{r}/IQ_covariances']) == len(self[f'{r}/corr_matrix']))
            except AssertionError:
                print(f'ProcessManager: The shapes of IQ matrices in {r} are not compatible with its readout_levels. '
                      +'New matrices will be generated. Please save it by calling cfg.save()')
                
                self[f'{r}/corr_matrix'] = np.identity(self[f'{r}/n_readout_levels']).tolist()
                self[f'{r}/IQ_covariances'] = [1 for i in range(self[f'{r}/n_readout_levels'])]
                self[f'{r}/IQ_means'] = [[i,i] for i in range(self[f'{r}/n_readout_levels'])]
                
                
    def process_data(self, measurement: dict, fitmodel: Model):
        """
        Process the data by performing rotation, average, GMM, fit, plot, etc.
        Three common routine are hard coded here since we never change them.
        User can define new routine by adding new key in process.yaml and add code here.
        
        Note from Zihao(02/17/2023):
        The new key should better be the first 'if' condition below.
        Because one may need to keep heralding to be true to add that pulse in sequence,
        while going into the customized process routine.
        """
        # r is 'R3', 'R4', data_dict has key 'Heterodyned_readout', etc.
        for r, data_dict in self.measurement.items():   
            if self['customized']:
                continue
            
            elif self['heralding']:
                continue
                
            elif self['classification']:
                data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Heterodyned_readout'], 
                                                           angle=self[f'{r}/IQ_rotation_angle'])
                data_dict['GMMpredicted_readout'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                                means=self[f'{r}/IQ_means'], 
                                                                covariances=self[f'{r}/IQ_covariances'])
                data_dict['PopulationNormalized_readout'] = normalize_population(data_dict['GMMpredicted_readout'],
                                                                                 n_levels=self[f'{r}/n_readout_levels'])
                data_dict['PopulationCorrected_readout'] = np.linalg.solve(self[f'{r}/corr_matrix'],
                                                                           data_dict['PopulationNormalized_readout'])
                # TODO: Figure out the last step, who times who.
                # TODO: think about how we do fit. Need to consider 2D data and multiple qubit scan.
                # Maybe it really worth to have the fit and plot outside the process_data.
                # Especially we actually have multiple level to fit.
            else:
                continue
        
        
        