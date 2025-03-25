"Living as realist, dreaming as idealist, always be curious, and never give up. --Z"

import os
from qtrlb.calibration import *
from qtrlb.benchmark import *
from qtrlb.processing import *
from qtrlb.projects import *
from qtrlb.config import *
from qtrlb.instruments import *




def begin_measurement_session(working_dir: str, 
                              variable_suffix: str = '',
                              test_mode: bool = False) -> MetaManager:
    """
    Instantiate all managers along with MetaManager, then load them.
    Return the instance of MetaManager.
    """
    yamls_path = os.path.join(working_dir, 'Yamls')
    
    varman = VariableManager(yamls_path, variable_suffix)
    dacman = DACManager(yamls_path, varman, test_mode)
    processman = ProcessManager(yamls_path, varman)
    dataman = DataManager(yamls_path, varman)
    instman = InstrumentManager(yamls_path, varman, test_mode)
    gateman = GateManager(yamls_path, varman)
    
    cfg = MetaManager(
        manager_dict={
            'variables': varman,
            'DAC': dacman,
            'process': processman,
            'data': dataman,
            'instruments': instman,           
            'gates': gateman
        },
        working_dir=working_dir
    )
    return cfg