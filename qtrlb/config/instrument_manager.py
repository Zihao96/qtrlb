import qtrlb.instruments as instruments
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager




class InstrumentManager(Config):
    """ This is a thin wrapper over the Config class to help with instruments management.
        The load() method will be called once in its __init__.
        This manager can be used independent of VariableManager by pass in varman=None.
        It will be useful for experiments outside of qtrl.calibration framework.
    
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
            varman: A VariableManager.
            test_mode: When true, you can run the whole program without a real instrument.
    """
    def __init__(self, 
                 yamls_path: str, 
                 varman: VariableManager = None,
                 test_mode: bool = False):
        super().__init__(yamls_path=yamls_path, 
                         suffix='instruments',
                         varman=varman)
        self.load()
        self.test_mode = test_mode
        if not test_mode: self.connect_instruments()


    def connect_instruments(self):
        """
        Connect all instruments listed in the yamls.
        """
        for name, instrument_dict in self.items():
            if not instrument_dict['connected']: continue
            kwargs = {k: v for k, v in instrument_dict.items() if k not in ('connected', 'device')}

            try: 
                inst = getattr(instruments, instrument_dict['device'])(**kwargs)
                setattr(self, name, inst)
            except Exception:
                print(f'Failed to connect {name}: {instrument_dict["device"]}.')

