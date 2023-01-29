from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager
from qblox_instruments import Cluster


class DACManager(Config):
    """ This is a thin wrapper over the Config class to help with DAC management.
        The load() method will be called once in its __init__.
    
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
            varman: A VariableManager.
    """
    def __init__(self, yamls_path: str, varman: VariableManager = None):
        super().__init__(yamls_path=yamls_path, 
                         suffix='DAC',
                         varman=varman)
        self.load()
    
    
    def load(self):
        """
        Run the parent load, then connect and reset the Qblox.
        """
        super().load()
        
        # Close any existing connections to any Cluster. Connect. Reset.
        Cluster.close_all()
        self.qblox = Cluster(self['name'], self['address'])
        self.qblox.reset()
























