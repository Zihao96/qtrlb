import os
from qtrlb.config.variable_manager import VariableManager
from qtrlb.config.DAC_manager import DACManager
from qtrlb.config.process_manager import ProcessManager
from qtrlb.config.data_manager import DataManager


def begin_measurement_session(working_dir: str, variable_suffix: str = ''):
    """
    Instantiate all managers along with MetaManager, then load them.
    Return the instance of MetaManager.
    """
    yamls_path = os.path.join(working_dir, 'Yamls')
    
    varman = VariableManager(yamls_path, variable_suffix)
    dacman = DACManager(yamls_path, varman)
    processman = ProcessManager(yamls_path, varman)
    dataman = DataManager(yamls_path, varman)
    
    cfg = MetaManager(manager_dict={'variables':varman,
                                    'DAC':dacman,
                                    'process':processman,
                                    'data':dataman},
                      working_dir=working_dir)
    return cfg


class MetaManager:
    """ A container for all managers to be accessible through one object.
        Each of their name will become an attribute of MetaManager.
        It also bring high-level interface for load, save, get, set, delete.
    
        Example of manager_dict:
        manager_dict = {'variables': varman,
                        'DAC': dacman,
                        'process': processman,
                        'data':dataman}
    """
    def __init__(self, manager_dict: dict, working_dir: str):
        self.manager_dict = manager_dict
        self.working_dir = working_dir
        
        for manager_name, manager in self.manager_dict.items():
            self.__setattr__(manager_name, manager)
      
    
    def load(self):
        """
        Call all managers' load method, VariableManager first.
        """
        self.variables.load()
        for manager in self.manager_dict.values():
            manager.load()  
        # It's fine we load VariableManager twice. It's fast.
        
        
    def save(self, yamls_path: str = None, verbose: bool = True):
        """
        Call all managers' save method.
        Allow user to pass a path of directory to save config_raw at another place.
        """
        for manager in self.manager_dict.values():
            manager.save(yamls_path=yamls_path, verbose=verbose)
    
        
    def get(self, key: str, which: str = 'dict'):
        """
        Get the value in the config_dict or config_raw corresponds to a given key.
        
        Examples: Suppose variable 'cfg' is a MetaManager.
            cfg['DAC.Module2/out0_att']
            cfg['process.R0/IQ_covariances']

        Parameters:
            key: str. String of keys separated by one dot and some forward slashes.
            which : str. Choose which dictionary we look up. Should only be 'dict' or 'raw'.
        """
        manager_name, key = key.split('.')
        manager = getattr(self, manager_name)
        return manager.get(key, which)
    
    
    def keys(self, key: str = '', which: str = 'dict'):
        """
        Get the keys of a dictionary inside the config_dict or config_raw corresponds to a given key.

        Parameters:
            key: str. String of keys separated by one dot and some forward slashes.
            which : str. Choose which dictionary we look up. Should only be 'dict' or 'raw'.
        """
        manager_name, key = key.split('.')
        manager = getattr(self, manager_name)
        return manager.keys(key, which)
        
        
    def set(self, key: str, value: type, which: str = 'both', save_raw: bool = False):
        """
        Set the value in the config_dict (and config_raw if asked) corresponds to a given key.
        Then save it if asked.
        If there is no corresponding key, a tree of dict will be built.

        Parameters
            key : str. String of keys separated by forward slash.
            value : TYPE. Value we want to set to given key.
            which : str. Choose which dictionary we look up. Should only be 'dict' or 'both'.
            save_raw : bool. Whether we want to save config_raw or not.
        """
        manager_name, key = key.split('.')
        manager = getattr(self, manager_name)
        return manager.set(key, value, which, save_raw)
        
    
    def delete(self, key: str, save_raw: bool = False):
        """
        Delete the item in the config_dict and config_raw corresponds to a given key.
        Then save it if asked.

        Parameters
            key : str. String of keys separated by forward slash.
            save_raw : bool. Whether we want to save config_raw or not.
        """
        manager_name, key = key.split('.')
        manager = getattr(self, manager_name)
        return manager.delete(key, save_raw)
    
    
    def __getitem__(self, key):
        return self.get(key)
        
        
    def __setitem__(self, key, value):
        return self.set(key, value)
        
    
    def __delitem__(self, key):
        return self.delete(key)
        
        
