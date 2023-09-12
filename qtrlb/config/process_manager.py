import numpy as np
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager
from qtrlb.processing.processing import rotate_IQ, gmm_predict, normalize_population, autorotate_IQ, \
                                        correct_population, two_tone_predict, two_tone_normalize


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
            self.set(f'{r}/lowest_readout_levels', self[f'{r}/readout_levels'][0], which='dict')
            self.set(f'{r}/highest_readout_levels', self[f'{r}/readout_levels'][-1], which='dict')
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
                
                
    def process_data(self, measurement: dict, shape: tuple, process_kwargs: dict = None):
        """
        Process the data by performing reshape, rotation, average, GMM, fit, plot, etc.
        Three common routine are hard coded here since we never change them.
        User can define new routine by adding function to ProcessManager and use it in variables.yaml.
        The shape is usually (2, n_reps, x_points) for base Scan and (2, n_reps, y_points, x_points) for 2D Scan.
        !*-*This function will change measurement in-place*-*!
        
        Note from Zihao(02/17/2023):
        The customized key should better be the first 'if' condition below.
        Because one may need to keep heralding to be true to add that pulse in sequence,
        while going into the customized process routine.
        """

        if self['customized'] is not None:
            getattr(self, self['customized'])(measurement, shape, process_kwargs)
        
        
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
                                                                 covariances=self[f'{r}/IQ_covariances'],
                                                                 lowest_level=self[f'{r}/lowest_readout_levels'])
                
                data_dict['GMMpredicted_heralding'] = gmm_predict(data_dict['IQrotated_heralding'], 
                                                                   means=self[f'{r}/IQ_means'], 
                                                                   covariances=self[f'{r}/IQ_covariances'],
                                                                   lowest_level=self[f'{r}/lowest_readout_levels'])
                
            heralding_mask = self.heralding_test(measurement=measurement)
            
            for r, data_dict in measurement.items(): 
                data_dict['Mask_heralding'] = heralding_mask  # So that it can be save to hdf5.
                
                data_dict['PopulationNormalized_readout'] = normalize_population(data_dict['GMMpredicted_readout'],
                                                                                 levels=self[f'{r}/readout_levels'],
                                                                                 mask=heralding_mask)
                
                data_dict['PopulationCorrected_readout'] = correct_population(data_dict['PopulationNormalized_readout'],
                                                                              self[f'{r}/corr_matrix'],
                                                                              self['corr_method'])
                
                data_dict['to_fit'] = data_dict['PopulationCorrected_readout']
            
            
        elif self['classification']:
            for r, data_dict in measurement.items():  
                data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
                
                data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Reshaped_readout'], 
                                                           angle=self[f'{r}/IQ_rotation_angle'])
                
                data_dict['GMMpredicted_readout'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                                means=self[f'{r}/IQ_means'], 
                                                                covariances=self[f'{r}/IQ_covariances'],
                                                                lowest_level=self[f'{r}/lowest_readout_levels'])
                
                data_dict['PopulationNormalized_readout'] = normalize_population(data_dict['GMMpredicted_readout'],
                                                                                 levels=self[f'{r}/readout_levels'])
                
                data_dict['PopulationCorrected_readout'] = correct_population(data_dict['PopulationNormalized_readout'],
                                                                              self[f'{r}/corr_matrix'],
                                                                              self['corr_method'])
                
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


    ##################################################           
    # All functions below are different customized data processing.
    # Use them by change customized_data_process in variables.yaml


    def two_tone_readout_mask(self, measurement: dict, shape: tuple, process_kwargs: dict):
        """
        Using two tones frequency-multiplexing to readout one resonator.
        Generate exactly same 'to_fit' value for both tones to avoid later bug.
        
        Example of process_kwargs:
        {
            ('R0', 'R1'): corr_matrix,
            ('R2', 'R3'): corr_matrix,
        }

        Note from Zihao(2023/06/21):
        I know it won't be elegant, but I also don't want to do premature optimization.
        Let's make it better when we really need to. 
        I didn't use ** on these kwargs. I believe dict bring us better encapsulation.
        Pass in a whole dictionary can be more general than just pass 'something = somevalue'.
        It's because the key doen't need to be string anymore.
        """
        # Normal GMM prediction as classification.
        for r, data_dict in measurement.items():  
            data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
            
            data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Reshaped_readout'], 
                                                       angle=self[f'{r}/IQ_rotation_angle'])
            
            data_dict['GMMpredicted_readout'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                             means=self[f'{r}/IQ_means'], 
                                                             covariances=self[f'{r}/IQ_covariances'],
                                                             lowest_level=self[f'{r}/lowest_readout_levels'])
            # GMMpredicted has shape (n_reps, y_points, x_points) for 2D Scan.
            # Values are integers as state assignment result.


        for (tone_0, tone_1), corr_matrix in process_kwargs.items():
            twotonepredicted_readout, mask_twotone = two_tone_predict(measurement[tone_0]['GMMpredicted_readout'],
                                                                      measurement[tone_1]['GMMpredicted_readout'],
                                                                      self[f'{tone_0}/readout_levels'],
                                                                      self[f'{tone_1}/readout_levels'])
            population_normalized_readout = normalize_population(twotonepredicted_readout,
                                                                 levels=np.union1d(self[f'{tone_0}/readout_levels'],
                                                                                   self[f'{tone_1}/readout_levels']),
                                                                 mask=mask_twotone)
            population_corrected_readout = correct_population(population_normalized_readout, corr_matrix)

            measurement[tone_0]['Mask_twotone'] = mask_twotone
            measurement[tone_1]['Mask_twotone'] = mask_twotone
            measurement[tone_0]['TwoTonepredicted_readout'] = twotonepredicted_readout
            measurement[tone_1]['TwoTonepredicted_readout'] = twotonepredicted_readout
            measurement[tone_0]['PopulationNormalized_readout'] = population_normalized_readout
            measurement[tone_1]['PopulationNormalized_readout'] = population_normalized_readout
            measurement[tone_0]['PopulationCorrected_readout'] = population_corrected_readout
            measurement[tone_1]['PopulationCorrected_readout'] = population_corrected_readout
            measurement[tone_0]['to_fit'] = population_corrected_readout
            measurement[tone_1]['to_fit'] = population_corrected_readout


    def two_tone_readout_corr(self, measurement: dict, shape: tuple, process_kwargs: dict):
        """
        Using two tones frequency-multiplexing to readout one resonator.
        Generate exactly same 'to_fit' value for both tones to avoid later bug.
        Require first resonator to have lower readout levels.
        
        Example of process_kwargs:
        {
            ('R0', 'R1'): corr_matrix,
            ('R2', 'R3'): corr_matrix,
        }

        Note from Zihao(2023/07/19):
        This method won't care about single shot result.
        Instead, it applies the correction matrix directly on the normalized population of each result pair.
        For example, if we first tone readout 0-3, second tone readout 3-6. The possible result are
        ((0,3),(0,4),(0,5),(0,6),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,3),(3,4),(3,5),(3,6))
        The corr_matrix should be 16*7, applied on actual 7 population, gives assigned 16 population.
        The mapping rule in this example is: (0-15) = 1 * (x1 - 3) + 4 * x0
        See two_tone_normalize for more details.
        """
        # Normal GMM prediction as classification.
        for r, data_dict in measurement.items():  
            data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
            
            data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Reshaped_readout'], 
                                                       angle=self[f'{r}/IQ_rotation_angle'])
            
            data_dict['GMMpredicted_readout'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                             means=self[f'{r}/IQ_means'], 
                                                             covariances=self[f'{r}/IQ_covariances'],
                                                             lowest_level=self[f'{r}/lowest_readout_levels'])
            # GMMpredicted has shape (n_reps, y_points, x_points) for 2D Scan.
            # Values are integers as state assignment result.


        for (tone_0, tone_1), corr_matrix in process_kwargs.items():
            twotonenormalized_readout = two_tone_normalize(measurement[tone_0]['GMMpredicted_readout'],
                                                           measurement[tone_1]['GMMpredicted_readout'],
                                                           self[f'{tone_0}/readout_levels'],
                                                           self[f'{tone_1}/readout_levels'])

            population_corrected_readout = correct_population(twotonenormalized_readout, corr_matrix)

            measurement[tone_0]['PopulationNormalized_readout'] = twotonenormalized_readout
            measurement[tone_1]['PopulationNormalized_readout'] = twotonenormalized_readout
            measurement[tone_0]['PopulationCorrected_readout'] = population_corrected_readout
            measurement[tone_1]['PopulationCorrected_readout'] = population_corrected_readout
            measurement[tone_0]['to_fit'] = population_corrected_readout
            measurement[tone_1]['to_fit'] = population_corrected_readout  


    def two_tone_readout_forc(self, measurement: dict, shape: tuple, process_kwargs: dict):
        raise NotImplementedError("ProcessManager: I'm Zihao. I'm sorry.")         