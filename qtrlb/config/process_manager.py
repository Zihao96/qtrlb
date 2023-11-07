import numpy as np
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager
from qtrlb.processing.processing import rotate_IQ, gmm_predict, heralding_test, normalize_population, \
    autorotate_IQ, correct_population, two_tone_predict, two_tone_normalize, multitone_predict_sequential, \
    multitone_predict_mask, multitone_normalize


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
        self.check_IQ_matrices()
        
        
    def check_IQ_matrices(self):
        """
        Check the shape of those IQ matrices inside process.yaml. 
        If their shapes are not compatible with readout_levels,
        default compatible matrices will be generated without saving.
        """
        for r in self.keys():
            if not r.startswith('R'): continue

            self.set(f'{r}/lowest_readout_levels', self[f'{r}/readout_levels'][0], which='dict')
            self.set(f'{r}/highest_readout_levels', self[f'{r}/readout_levels'][-1], which='dict')
            self.set(f'{r}/n_readout_levels', len(self[f'{r}/readout_levels']), which='dict')
            
            if not (self[f'{r}/n_readout_levels'] == len(self[f'{r}/IQ_means']) 
                    == len(self[f'{r}/IQ_covariances']) == len(self[f'{r}/corr_matrix'])):
                
                print(f'Processman: New IQ matrices of {r} has been generated to be compatible with its readout_levels.')
                self[f'{r}/corr_matrix'] = np.identity(self[f'{r}/n_readout_levels'])
                self[f'{r}/IQ_covariances'] = [1 for _ in range(self[f'{r}/n_readout_levels'])]
                self[f'{r}/IQ_means'] = [[i, i] for i in range(self[f'{r}/n_readout_levels'])]
                
                
    def process_data(self, measurement: dict, shape: tuple, process_kwargs: dict = None):
        """
        Process the data by performing reshape, rotation, average, GMM, fit, plot, etc.
        Three common routine are hard coded here since we never change them.
        User can define new routine by adding function to ProcessManager and use it in variables.yaml.
        The shape is usually (2, n_reps, x_points) for base Scan and (2, n_reps, y_points, x_points) for 2D Scan.
        !*-*This function will change measurement in-place*-*!
        
        Note from Zihao(02/17/2023):
        The customized_data_process key should better be the first 'if' condition below.
        Because one may need to keep heralding to be true to add that pulse in sequence,
        while going into the customized process routine.
        """

        if self['customized_data_process'] is not None:
            getattr(self, self['customized_data_process'])(measurement, shape, process_kwargs)
        
        
        elif self['heralding'] is True:
            # r is 'R3', 'R4', data_dict has key 'Heterodyned_readout', etc.
            heralding_data_list = []
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
                heralding_data_list.append(data_dict['GMMpredicted_heralding'])

            mask_heralding = heralding_test(*heralding_data_list)
            
            for r, data_dict in measurement.items(): 
                data_dict['Mask_heralding'] = mask_heralding  # So that it can be save to hdf5.
                
                data_dict['PopulationNormalized_readout'] = normalize_population(data_dict['GMMpredicted_readout'],
                                                                                 levels=self[f'{r}/readout_levels'],
                                                                                 mask=mask_heralding)
                
                data_dict['PopulationCorrected_readout'] = correct_population(data_dict['PopulationNormalized_readout'],
                                                                              self[f'{r}/corr_matrix'],
                                                                              self['corr_method'])
                
                data_dict['to_fit'] = data_dict['PopulationCorrected_readout']
            
            
        elif self['classification'] is True:
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
            population_corrected_readout = correct_population(population_normalized_readout, 
                                                              corr_matrix, 
                                                              self['corr_method'])

            measurement[tone_0]['Mask_twotone'] = mask_twotone
            measurement[tone_1]['Mask_twotone'] = mask_twotone
            measurement[tone_0]['TwoTonePredicted_readout'] = twotonepredicted_readout
            measurement[tone_1]['TwoTonePredicted_readout'] = twotonepredicted_readout
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

            population_corrected_readout = correct_population(twotonenormalized_readout, 
                                                              corr_matrix, 
                                                              self['corr_method'])

            measurement[tone_0]['PopulationNormalized_readout'] = twotonenormalized_readout
            measurement[tone_1]['PopulationNormalized_readout'] = twotonenormalized_readout
            measurement[tone_0]['PopulationCorrected_readout'] = population_corrected_readout
            measurement[tone_1]['PopulationCorrected_readout'] = population_corrected_readout
            measurement[tone_0]['to_fit'] = population_corrected_readout
            measurement[tone_1]['to_fit'] = population_corrected_readout  


    def multitone_readout_sequential(self, measurement: dict, shape: tuple, process_kwargs: dict):
        """
        Using multi-tones frequency-multiplexing to readout each resonator.
        Generate exactly same 'to_fit' value for each tones to avoid later bug.
        
        Example of process_kwargs:
        {
            ('R0a', 'R0b'): corr_matrix,
            ('R4a', 'R4b', 'R4c'): corr_matrix,
        }
        Here three tone of 'R4' MUST be in ascending order, meaning 'R4a' reading lowest level.
        
        About this readout strategy:
        In this sequential method, we will always trust first tone.
        Only when first tone gives its highest result, we will look at the second tone, and so on.
        It works well when even higher states, which is higher than the highest possible state of this tone, 
        won't change IQ Gaussian out of the highest Gaussian and hence won't have too much miss classification.
        Pros: all data are used, always have single shot result, support realtime feedback.
        Cons: result might be slightly biased to lower level because the problem mentioned above.
        It usually works better we have further Gaussian separation and high single tone readout fidelity. 
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

        # All heralding process in this if condition.
        mask_heralding = None
        if self['heralding'] is True:
            for r, data_dict in measurement.items():  
                data_dict['Reshaped_heralding'] = np.array(data_dict['Heterodyned_heralding']).reshape(shape)
                data_dict['IQrotated_heralding'] = rotate_IQ(data_dict['Reshaped_heralding'], 
                                                             angle=self[f'{r}/IQ_rotation_angle'])           
                data_dict['GMMpredicted_heralding'] = gmm_predict(data_dict['IQrotated_heralding'], 
                                                                  means=self[f'{r}/IQ_means'], 
                                                                  covariances=self[f'{r}/IQ_covariances'],
                                                                  lowest_level=self[f'{r}/lowest_readout_levels'])

            # For each resonator, we only use it's first tone for heralding test.
            # This is equivalent to multitone readout for sequential method.
            heralding_data_list = [measurement[tones[0]]['GMMpredicted_heralding'] for tones in process_kwargs.keys()]
            mask_heralding = heralding_test(*heralding_data_list)
            

        # Multitone process.
        for tones, corr_matrix in process_kwargs.items():
            data_levels_tuple = (
                (measurement[tone]['GMMpredicted_readout'], self[f'{tone}/readout_levels']) for tone in tones
            )
            levels = np.arange(self[f'{tones[0]}/readout_levels'][0], 
                               self[f'{tones[-1]}/readout_levels'][-1] + 1,
                               step=1, dtype=int)

            multitonepredicted_readout = multitone_predict_sequential(*data_levels_tuple)
            population_normalized_readout = normalize_population(multitonepredicted_readout, 
                                                                 levels,
                                                                 mask=mask_heralding)
            population_corrected_readout = correct_population(population_normalized_readout, 
                                                              corr_matrix, 
                                                              self['corr_method'])

            for tone in tones:
                measurement[tone]['Mask_heralding'] = mask_heralding
                measurement[tone]['MultiTonePredicted_readout'] = multitonepredicted_readout
                measurement[tone]['PopulationNormalized_readout'] = population_normalized_readout
                measurement[tone]['PopulationCorrected_readout'] = population_corrected_readout
                measurement[tone]['to_fit'] = population_corrected_readout


    def multitone_readout_mask(self, measurement: dict, shape: tuple, process_kwargs: dict):
        """
        Using multi-tones frequency-multiplexing to readout each resonator.
        Generate exactly same 'to_fit' value for each tones to avoid later bug.
        
        Example of process_kwargs:
        {
            ('R0a', 'R0b'): corr_matrix,
            ('R4a', 'R4b', 'R4c'): corr_matrix,
        }
        Here three tone of 'R4' MUST be in ascending order, meaning 'R4a' reading lowest level.
        
        About this readout strategy:
        In this mask method, we will do normalization with a mask.
        The data will be masked when there is contradiction.
        It works better than sequential and MLE since the data being masked may experience decoherence during readout.
        Pros: Uncorrected result should have less bias. Slightly higher readout fidelity and easy to implement.
        Cons: The single shot is not guaranteed.
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

        # Multitone process readout first.
        for tones, corr_matrix in process_kwargs.items():
            data_levels_tuple = (
                (measurement[tone]['GMMpredicted_readout'], self[f'{tone}/readout_levels']) for tone in tones
            )
            levels = np.arange(self[f'{tones[0]}/readout_levels'][0], 
                               self[f'{tones[-1]}/readout_levels'][-1] + 1,
                               step=1, dtype=int)

            multitonepredicted_readout, mask_multitone_readout = multitone_predict_mask(*data_levels_tuple)
            population_normalized_readout = normalize_population(multitonepredicted_readout, 
                                                                 levels,
                                                                 mask=mask_multitone_readout)
            population_corrected_readout = correct_population(population_normalized_readout, 
                                                              corr_matrix, 
                                                              self['corr_method'])

            # Save to measurement
            for tone in tones:
                measurement[tone]['MultiTonePredicted_readout'] = multitonepredicted_readout
                measurement[tone]['Mask_multitone_readout'] = mask_multitone_readout
                measurement[tone]['PopulationNormalized_readout'] = population_normalized_readout
                measurement[tone]['PopulationCorrected_readout'] = population_corrected_readout
                measurement[tone]['to_fit'] = population_corrected_readout


        # All heralding process in this if condition.
        # It will overwrite some of the previous data and doesn't sounds efficient.
        # But I believe it make code easy to read and is not the performance bottleneck yet.
        if self['heralding'] is True:
            for r, data_dict in measurement.items():  
                data_dict['Reshaped_heralding'] = np.array(data_dict['Heterodyned_heralding']).reshape(shape)
                data_dict['IQrotated_heralding'] = rotate_IQ(data_dict['Reshaped_heralding'], 
                                                             angle=self[f'{r}/IQ_rotation_angle'])           
                data_dict['GMMpredicted_heralding'] = gmm_predict(data_dict['IQrotated_heralding'], 
                                                                  means=self[f'{r}/IQ_means'], 
                                                                  covariances=self[f'{r}/IQ_covariances'],
                                                                  lowest_level=self[f'{r}/lowest_readout_levels'])

            # Multitone process for heralding.
            # In this method, we also use mask strategy on heralding measurement.
            # It means we need to do multitone process first before we do heralding test.
            for tones in process_kwargs.keys():
                data_levels_tuple = (
                    (measurement[tone]['GMMpredicted_heralding'], self[f'{tone}/readout_levels']) for tone in tones
                )
                multitonepredicted_heralding, mask_multitone_heralding = multitone_predict_mask(*data_levels_tuple)

                # Save to measurement
                for tone in tones:
                    measurement[tone]['MultiTonePredicted_heralding'] = multitonepredicted_heralding
                    measurement[tone]['Mask_multitone_heralding'] = mask_multitone_heralding

            # Heralding test. We don't trim since we won't have trimed result anyway.
            heralding_data_list = [measurement[tones[0]]['MultiTonePredicted_heralding'] for tones in process_kwargs.keys()]
            mask_heralding = heralding_test(*heralding_data_list, trim=False)

            for tones, corr_matrix in process_kwargs.items():
                mask_union = measurement[tones[0]]['Mask_multitone_heralding'] | \
                    mask_heralding | measurement[tones[0]]['Mask_multitone_readout']

                population_normalized_readout = normalize_population(measurement[tones[0]]['MultiTonePredicted_readout'], 
                                                                     levels,
                                                                     mask=mask_union)
                population_corrected_readout = correct_population(population_normalized_readout, 
                                                                  corr_matrix, 
                                                                  self['corr_method'])

                # Save to measurement
                for tone in tones:
                    measurement[tone]['Mask_heralding'] = mask_heralding
                    measurement[tone]['Mask_union'] = mask_union
                    measurement[tone]['PopulationNormalized_readout'] = population_normalized_readout
                    measurement[tone]['PopulationCorrected_readout'] = population_corrected_readout
                    measurement[tone]['to_fit'] = population_corrected_readout


    def multitone_readout_corr(self, measurement: dict, shape: tuple, process_kwargs: dict):
        """
        Using multi-tones frequency-multiplexing to readout each resonator.
        Generate exactly same 'to_fit' value for each tones to avoid later bug.
        
        Example of process_kwargs:
        {
            ('R0a', 'R0b'): corr_matrix,
            ('R4a', 'R4b', 'R4c'): corr_matrix,
        }
        Here three tone of 'R4' MUST be in ascending order, meaning 'R4a' reading lowest level.
        
        About this readout strategy:
        In this correction method, we don't do any multitone prediction.
        It means we won't generate a final state assignment for each single shot.
        We will keep the prediction from each tone and directly apply readout correction on it.
        We map each combination of these prediction to be a meaningless integer.
        Mapping rule: A + len(A) * (B - intersection_AB) + len(A) * len(B) * (C - intersection_BC) + ...
        Pros: If we compare post-correction fidelity, this one should win a little and possibly more stable.
        Cons: Doesn't have any single shot at all. Only have a normalized-corrected population/probability.
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

        # All heralding process in this if condition.
        mask_heralding = None
        if self['heralding'] is True:
            for r, data_dict in measurement.items():  
                data_dict['Reshaped_heralding'] = np.array(data_dict['Heterodyned_heralding']).reshape(shape)
                data_dict['IQrotated_heralding'] = rotate_IQ(data_dict['Reshaped_heralding'], 
                                                             angle=self[f'{r}/IQ_rotation_angle'])           
                data_dict['GMMpredicted_heralding'] = gmm_predict(data_dict['IQrotated_heralding'], 
                                                                  means=self[f'{r}/IQ_means'], 
                                                                  covariances=self[f'{r}/IQ_covariances'],
                                                                  lowest_level=self[f'{r}/lowest_readout_levels'])

            # For each resonator, we only use it's first tone for heralding test.
            # Until here, everything is same as sequential method.
            heralding_data_list = [measurement[tones[0]]['GMMpredicted_heralding'] for tones in process_kwargs.keys()]
            mask_heralding = heralding_test(*heralding_data_list)
            

        # Multitone process.
        for tones, corr_matrix in process_kwargs.items():
            data_levels_tuple = (
                (measurement[tone]['GMMpredicted_readout'], self[f'{tone}/readout_levels']) for tone in tones
            )
            population_normalized_readout = multitone_normalize(*data_levels_tuple, mask=mask_heralding)
            population_corrected_readout = correct_population(population_normalized_readout, 
                                                              corr_matrix, 
                                                              self['corr_method'])

            for tone in tones:
                measurement[tone]['Mask_heralding'] = mask_heralding
                measurement[tone]['PopulationNormalized_readout'] = population_normalized_readout
                measurement[tone]['PopulationCorrected_readout'] = population_corrected_readout
                measurement[tone]['to_fit'] = population_corrected_readout

