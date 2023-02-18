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
        
        
    def save(self, yamls_path=None):
        """
        Call all managers' save method.
        Allow user to pass a path of directory to save config_raw at another place.
        """
        for manager in self.manager_dict.values():
            manager.save(yamls_path=yamls_path)
        
        
        
        
