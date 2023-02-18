import os
from qtrlb.config.variable_manager import VariableManager
from qtrlb.config.DAC_manager import DACManager
from qtrlb.config.process_manager import ProcessManager
from qtrlb.config.data_manager import DataManager
from qtrlb.config.meta_manager import MetaManager



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
    
    
    
    
    
    