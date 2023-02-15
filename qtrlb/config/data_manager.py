import datetime
import os
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager
from qtrlb.config.meta_manager import MetaManager


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
                 varman: VariableManager):
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
        self.datetime = time if time is not None else datetime.datetime.now()
        
        self.date = self.datetime.strftime(self['date_fmt'])
        self.time = self.datetime.strftime(self['time_fmt'])
        self.basename = self.time + '_' + experiment_type + '_' + 'experiment_suffix'
        
        self.data_path = os.path.join(self['base_directory'], self.date, self.basename)
        self.yamls_path = os.path.join(self.data_path, 'Yamls')
        self.jsons_path = os.path.join(self.data_path, 'Jsons')
        
        try:
            os.makedirs(self.yamls_path)
            os.makedirs(self.jsons_path)
            return self.data_path
        except FileExistsError:
            print('DataManager: Experiment directory exists. No directory will be created.')
            return self.data_path
        
        
        
    
        

        