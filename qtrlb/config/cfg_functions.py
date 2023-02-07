from qtrlb.config.variable_manager import VariableManager
from qtrlb.config.DAC_manager import DACManager
from qtrlb.config.meta_manager import MetaManager



def begin_measurement_session(yamls_path: str, variable_suffix: str = ''):
    """
    Instantiate all managers along with MetaManager, then load them.
    Return the instance of MetaManager.
    """
    varman = VariableManager(yamls_path, variable_suffix)
    dacman = DACManager(yamls_path, varman)
    
    cfg = MetaManager({'variables':varman,
                       'DAC':dacman})
    
    
    
    
    
    
    