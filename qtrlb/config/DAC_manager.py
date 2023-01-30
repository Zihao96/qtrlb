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
    def __init__(self, 
                 yamls_path: str, 
                 varman: VariableManager = None):
        super().__init__(yamls_path=yamls_path, 
                         suffix='DAC',
                         varman=varman)
        Cluster.close_all()
        self.qblox = Cluster(self['name'], self['address'])
        self.qblox.reset()
        self.load()
    
    
    def load(self):
        """
        Run the parent load, then implement mixer calibration parameters.
        Because the parameter depends on AWG frequency of each sequencer,
        it's not a good idea to implement it here. --Zihao(01/29/2023)
        """
        super().load()
        
        modules_list = [key for key in self.keys() if key.startswith('module')]
        self.set('Modules', modules_list, which='dict')  # Keep the new key start with uppercase!
        
        self.implement_parameters()
    
    
    def implement_parameters(self):
        """
        The temporary way to null the mixer and load built-in attenuation.
        We suppose the parameters are independent of both LO and AWG frequency.
        I intentionally exclude the LO frequency here. --Zihao(01/29/2023)
        """
        for module in self['Modules']:
            this_module = getattr(self.qblox, f'module{module}')
            
            for attribute in self[f'{module}'].keys():
                # Sorry, this is just temporary code.
                if attribute.startswith('out') or attribute.startswith('in') or attribute.startswith('scope'):
                    setattr(this_module, attribute, self[f'{module}/{attribute}'])
        
            for sequencer in this_module.sequencers:
                sequencer.mixer_corr_gain_ratio(self[f'{module}/mixer_corr_gain_ratio'])
                sequencer.mixer_corr_phase_offset_degree(self[f'{module}/mixer_corr_phase_offset_degree'])
    
    
    def disconnect_existed_map(self):
        """
        Disconnect existed maps between two output paths of each output port 
        and two output paths of each sequencer.
        """
        for module in self['Modules']:
            this_module = getattr(self.qblox, module)
            
            # Steal code from Qblox official tutoirals.
            if self[f'{module}/type'] == 'QCM-RF':
                for sequencer in this_module.sequencers:
                    for out in range(0, 4):
                        if hasattr(sequencer, "channel_map_path{}_out{}_en".format(out%2, out)):
                            sequencer.set("channel_map_path{}_out{}_en".format(out%2, out), False)
                    
            elif self[f'{module}/type'] == 'QRM-RF':
                for sequencer in this_module.sequencers:
                    for out in range(0, 2):
                        if hasattr(sequencer, "channel_map_path{}_out{}_en".format(out%2, out)):
                            sequencer.set("channel_map_path{}_out{}_en".format(out%2, out), False)
                
            else:
                raise ValueError(f'The type of {module} is invalid.')


    def build_channel_map(self):
        """
        Build maps between two output paths of each output port 
        and two output paths of each sequencer, based on Variable.yaml
        """
        
        # For drive
        for qubit in self.varman['qubits']:
            qubit_module = self.varman[f'{qubit}/module']
            qubit_sequencer = self.varman[f'{qubit}/sequencer']
            qubit_out = self.varman[f'{qubit}/out']

            qblox_module = getattr(self.qblox, f'module{qubit_module}')
            qblox_sequencer = getattr(qblox_module, f'sequencer{qubit_sequencer}')
            
            if qubit_out == 0:
                qblox_sequencer.channel_map_path0_out0_en(True)
                qblox_sequencer.channel_map_path1_out1_en(True)
            elif qubit_out == 1:
                qblox_sequencer.channel_map_path0_out2_en(True)
                qblox_sequencer.channel_map_path1_out3_en(True)
            else:
                raise ValueError(f'The value of out{qubit_out} in {qubit} is invalid.')

        # For readout
        readout_module = self.varman[f'{qubit}/module']
        qblox_module = getattr(self.qblox, f'module{readout_module}')
        for sequencer in qblox_module.sequencers:
            sequencer.channel_map_path0_out0_en(True)
            sequencer.channel_map_path1_out1_en(True)















