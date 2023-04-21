import numpy as np
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager
from qtrlb.processing.processing import rotate_IQ, gmm_predict, normalize_population, autorotate_IQ, correct_population


class ProcessManager(Config):
    """ This is a thin wrapper over the Config class to help with measurement process management.
        The load() method will be called once in its __init__.
    
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
    """
    def __init__(self, 
                 yamls_path: str, 
                 varman: VariableManager = None):
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
                      +'New matrices has been generated. Please save it by calling cfg.save()')
                
                self[f'{r}/corr_matrix'] = np.identity(self[f'{r}/n_readout_levels'])
                self[f'{r}/IQ_covariances'] = [1 for i in range(self[f'{r}/n_readout_levels'])]
                self[f'{r}/IQ_means'] = [[i*10, i*10] for i in range(self[f'{r}/n_readout_levels'])]
                
                
    def process_data(self, measurement: dict, shape: tuple):
        """
        Process the data by performing reshape, rotation, average, GMM, fit, plot, etc.
        Three common routine are hard coded here since we never change them.
        User can define new routine by adding new key in process.yaml and add code here.
        The shape is expected to be (2, n_reps, x_points).
        
        Note from Zihao(02/17/2023):
        The new key should better be the first 'if' condition below.
        Because one may need to keep heralding to be true to add that pulse in sequence,
        while going into the customized process routine.
        """

        if self['customized']:
            pass
        
        
        elif self['heralding']:
            # r is 'R3', 'R4', data_dict has key 'Heterodyned_readout', etc.
            for r, data_dict in measurement.items():  
                data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
                data_dict['Reshaped_heralding'] = np.array(data_dict['Heterodyned_heralding']).reshape(shape)
                
                data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Reshaped_readout'], 
                                                           angle=self[f'{r}/IQ_rotation_angle'])
                data_dict['IQrotated_heralding'] = rotate_IQ(data_dict['Reshaped_heralding'], 
                                                             angle=self[f'{r}/IQ_rotation_angle'])
                
                data_dict['GMMpredicted_readout'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                                means=self[f'{r}/IQ_means'], 
                                                                covariances=self[f'{r}/IQ_covariances'])
                data_dict['GMMpredicted_heralding'] = gmm_predict(data_dict['IQrotated_heralding'], 
                                                                  means=self[f'{r}/IQ_means'], 
                                                                  covariances=self[f'{r}/IQ_covariances'])
                
            heralding_mask = self.heralding_test(measurement=measurement)
            
            for r, data_dict in measurement.items(): 
                data_dict['Mask_heralding'] = heralding_mask  # So that it can be save to hdf5.
                
                data_dict['PopulationNormalized_readout'] = normalize_population(data_dict['GMMpredicted_readout'],
                                                                                 n_levels=self[f'{r}/n_readout_levels'],
                                                                                 mask=heralding_mask)
                
                data_dict['PopulationCorrected_readout'] = correct_population(data_dict['PopulationNormalized_readout'],
                                                                              self[f'{r}/corr_matrix'])
                
                data_dict['to_fit'] = data_dict['PopulationCorrected_readout']
            
            
        elif self['classification']:
            for r, data_dict in measurement.items():  
                data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
                
                data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Reshaped_readout'], 
                                                           angle=self[f'{r}/IQ_rotation_angle'])
                
                data_dict['GMMpredicted_readout'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                                means=self[f'{r}/IQ_means'], 
                                                                covariances=self[f'{r}/IQ_covariances'])
                
                data_dict['PopulationNormalized_readout'] = normalize_population(data_dict['GMMpredicted_readout'],
                                                                                 n_levels=self[f'{r}/n_readout_levels'])
                
                data_dict['PopulationCorrected_readout'] = correct_population(data_dict['PopulationNormalized_readout'],
                                                                              self[f'{r}/corr_matrix'])
                
                data_dict['to_fit'] = data_dict['PopulationCorrected_readout']

            
        else:
            for r, data_dict in measurement.items():  
                data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
                
                data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Reshaped_readout'], 
                                                           angle=self[f'{r}/IQ_rotation_angle'])
                
                data_dict['IQaveraged_readout'] = np.mean(data_dict['IQrotated_readout'], axis=1)
                
                if self['IQautorotation']:
                    data_dict['IQautorotated_readout'] = autorotate_IQ(data_dict['IQrotated_readout'], 
                                                                        n_components=self[f'{r}/n_readout_levels'])
                    data_dict['IQaveraged_readout'] = np.mean(data_dict['IQautorotated_readout'], axis=1)
        
                data_dict['to_fit'] = data_dict['IQaveraged_readout']
        
        
    @staticmethod
    def heralding_test(measurement: dict) -> np.ndarray:
        """
        Generate the ndarray heralding_mask with shape (n_reps, x_points).
        The entries will be 0 only if all resonators gives 0 in heralding.
        It means for that specific repetition and x_point, all resonators pass heralding test.
        We then truncate data to make sure all x_point has same amount of available repetition.
        
        Note from Zihao(02/21/2023):
        The code here is stolen from original version of qtrl where we can only test ground state.
        However, ground state has most population and if our experiment need to start from |1>, pi pulse it.
        The code here is ugly and hard to read, please make it better if you know how to do it.
        """
        resonators = list(measurement.keys())
        heralding_mask = np.zeros_like(measurement[resonators[0]]['GMMpredicted_heralding'])
        
        for r in resonators:
            heralding_mask = heralding_mask | measurement[r]['GMMpredicted_heralding']
            
        n_pass_min = np.min(np.sum(heralding_mask == 0, axis=0))  
        
        for i in range(heralding_mask.shape[1]): # Loop over each x_point
            j = 0
            while np.sum(heralding_mask[:, i] == 0) > n_pass_min:
                n_short = np.sum(heralding_mask[:, i] == 0) - n_pass_min
                heralding_mask[j : j + n_short, i] = -1
                j += n_short
                
        return heralding_mask



    
