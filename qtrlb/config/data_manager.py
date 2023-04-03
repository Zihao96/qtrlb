import datetime
import os
import h5py
import traceback
import numpy as np
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager


class DataManager(Config):
    """ This is a thin wrapper over the Config class to help with data management.
        The load() method will be called once in its __init__.
        This manager can be used independent of VariableManager by pass in varman=None.
        It will be useful for experiment outside of qtrl.calibration framework.
    
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
            varman: A VariableManager.
    """
    def __init__(self, 
                 yamls_path: str, 
                 varman: VariableManager = None):
        super().__init__(yamls_path=yamls_path, 
                         suffix='data',
                         varman=varman)
        self.load()


    def make_exp_dir(self, experiment_type: str, experiment_suffix: str, time: datetime.datetime = None):
        """
        Make a saving directory under self['base_directory'].
        Return to path: 'base_directory/date_fmt/time_fmt+experiment_type+experiment_suffix'.
        Notice the basename won't repeat here.
        Other function can get basename by calling os.path.basename(data_path)
        Overwriting is forbidden here.
        
        Example of Attribute:
            experiment_type: 'drive_amplitude', 't1', 'ramsey'
            experiment_suffix: 'LLR2_EJEC_Q4_AD+200kHz_happynewyear'
        """
        _chip_name = self.varman.variable_suffix if self.varman is not None else ''
        
        self.datetime = time if time is not None else datetime.datetime.now()
        date = self.datetime.strftime(self['date_fmt'])
        time = self.datetime.strftime(self['time_fmt'])
        
        self.datetime_stamp = '/'.join([date, time])
        self.basename = time + _chip_name + '_' + experiment_type + '_' + experiment_suffix
        
        self.data_path = os.path.join(self['base_directory'], date, self.basename)
        self.yamls_path = os.path.join(self.data_path, 'Yamls')
        self.jsons_path = os.path.join(self.data_path, 'Jsons')
        
        try:
            os.makedirs(self.yamls_path)
            os.makedirs(self.jsons_path)
        except FileExistsError:
            traceback_str = traceback.format_exc()
            print('DataManager: Experiment directory exists. No directory will be created.')
            print(traceback_str)
        
        
    @staticmethod
    def save_measurement(data_path: str, measurement: dict, attrs: dict = None):
        """
        Save the measurement dictionary into a hdf5.
        I keep this layer because we can also pass information of scan here and save it as attrs.
        """
        if attrs is None: attrs = {}
        hdf5_path = os.path.join(data_path, 'measurement.hdf5')  
        
        with h5py.File(hdf5_path, 'w') as h5file:
            DataManager.save_dict_to_hdf5(measurement, h5file)
            for k, v in attrs.items(): 
                try:
                    h5file.attrs[k] = v
                # Some attributes, cfg for instance, cannot (and shouldn't) be saved.
                except TypeError:  
                    pass
        
        
    @staticmethod
    def save_dict_to_hdf5(dictionary: dict, h5: h5py.File | h5py.Group):
        """
        Recursively save a nested dictionary to hdf5 file/group. 
        """
        for k, v in dictionary.items():
            if isinstance(v, dict):
                subgroup = h5.create_group(k)
                DataManager.save_dict_to_hdf5(v, subgroup)
            elif v is None:
                continue
            else:
                h5.create_dataset(k, data = v)

                

    @staticmethod                
    def load_hdf5_to_dict(h5: h5py.File | h5py.Group, dictionary: dict = None):
        """
        Recursively load a hdf5 file/group to a nested dictionary.
        """
        if dictionary == None: dictionary={}
        dictionary = dict(h5)
        
        for k, v in dictionary.items():
            if isinstance(v, h5py.Group):
                dictionary[k] = DataManager.load_hdf5_to_dict(v, dictionary[k])
            else:
                dictionary[k] = np.array(v)
        return dictionary