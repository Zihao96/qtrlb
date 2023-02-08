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
                 varman: VariableManager):
        super().__init__(yamls_path=yamls_path, 
                         suffix='DAC',
                         varman=varman)
        
        self.load()
        
        Cluster.close_all()
        self.qblox = Cluster(self['name'], self['address'])
        self.qblox.reset()
    
    
    def load(self):
        """
        Run the parent load and add necessary new item in config_dict.
        """
        super().load()
        
        modules_list = [key for key in self.keys() if key.startswith('Module')]
        self.set('modules', modules_list, which='dict')  # Keep the new key start with lowercase!
        
        
    def implement_parameters(self, qubits: list, resonators: list, subspace: list):
        """
        Implement the setting/parameters onto Qblox.
        This function should be called after we know which specific qubits/resonators will be used.
        The qubits/resonators should be list of string: ['Q2', 'Q4'], ['R1', 'R3'], etc.
        
        Right now it's just a temporary way to null the mixer.
        We suppose the parameters are independent of both LO and AWG frequency.
        In future we should have frequency-dependent mixer nulling. --Zihao(01/29/2023)
        """
        self.qblox.reset()
        self.disconnect_existed_map()
        
        # Qubits first, then resonators. Module first, then Sequencer.
        for i, q in enumerate(qubits):
            qubit_module = self.varman[f'{q}/module']  # Just an interger. It's for convenience.
            qubit_sequencer = self.varman[f'{q}/sequencer']
            this_module = getattr(self.qblox, f'module{qubit_module}')  # The real module/sequencer object.
            this_sequencer = getattr(this_module, f'sequencer{qubit_sequencer}')  
            
            for attribute in self[f'Module{qubit_module}'].keys():
                if attribute.startswith('out') or attribute.startswith('in'):
                    this_attribute = getattr(this_module, attribute)
                    this_attribute(self[f'Module{qubit_module}/{attribute}'])
            
            this_attribute = getattr(this_module, 'out{}_lo_freq'.format(self.varman[f'{q}/out']))
            this_attribute(self.varman[f'{q}/qubit_LO'])        
            this_sequencer.sync_en(True)
            this_sequencer.mod_en_awg(True)
            this_attribute = getattr(this_sequencer, 'channel_map_path0_out{}_en'.format(self.varman[f'{q}/out'] * 2))
            this_attribute(True)
            this_attribute = getattr(this_sequencer, 'channel_map_path1_out{}_en'.format(self.varman[f'{q}/out'] * 2 + 1))
            this_attribute(True)
            
            # this_sequencer.gain_awg_path0(self.varman[f'{q}/{subspace[i]}/amp_rabi'])
            # this_sequencer.gain_awg_path1(self.varman[f'{q}/{subspace[i]}/amp_rabi'])
            # this_sequencer.nco_freq(self.varman[f'{q}/{subspace[i]}/mod_freq']) 
            this_sequencer.mixer_corr_gain_ratio(self[f'Module{qubit_module}/mixer_corr_gain_ratio'])           
            this_sequencer.mixer_corr_phase_offset_degree(self[f'Module{qubit_module}/mixer_corr_phase_offset_degree'])
            this_sequencer.sequence(f'./Jsons/{q}_sequence.json')

        
        for r in resonators:
            resonator_module = self.varman[f'{r}/module']
            resonator_sequencer = self.varman[f'{r}/sequencer']
            this_module = getattr(self.qblox, f'module{resonator_module}')
            this_sequencer = getattr(this_module, f'sequencer{resonator_sequencer}')  
            
            for attribute in self[f'Module{resonator_module}'].keys():
                if attribute.startswith('out') or attribute.startswith('in') or attribute.startswith('scope'):
                    this_attribute = getattr(this_module, attribute)
                    this_attribute(self[f'Module{resonator_module}/{attribute}'])
            
            this_module.out0_in0_lo_freq(self.varman[f'{r}/resonator_LO'])        
            this_module.scope_acq_sequencer_select(self.varman[f'{r}/sequencer'])  # Last sequencer to triger acquire.
            this_sequencer.sync_en(True)
            this_sequencer.mod_en_awg(True)
            this_sequencer.demod_en_acq(True)
            this_sequencer.integration_length_acq(round(self.varman['common/integration_length'] * 1e9))
            this_sequencer.channel_map_path0_out0_en(True)
            this_sequencer.channel_map_path1_out1_en(True)

            # this_sequencer.gain_awg_path0(self.varman[f'{r}/amp'])
            # this_sequencer.gain_awg_path1(self.varman[f'{r}/amp'])
            # this_sequencer.nco_freq(self.varman[f'{r}/mod_freq'])                    
            this_sequencer.mixer_corr_gain_ratio(self[f'Module{resonator_module}/mixer_corr_gain_ratio'])
            this_sequencer.mixer_corr_phase_offset_degree(self[f'Module{resonator_module}/mixer_corr_phase_offset_degree'])
            this_sequencer.sequence(f'./Jsons/{r}_sequence.json')
            
        # Sorry, this is just temporary code. Maybe I should use the replace_vars trick here.
        # But it's tricky to set those freq/amp based on which qubit, and we may have the previous pulse.yaml problem. 
        # --Zihao (01/30/2023)
    
    
    def disconnect_existed_map(self):
        """
        Disconnect all existed maps between two output paths of each output port 
        and two output paths of each sequencer.
        """
        for m in self['modules']:
            this_module = getattr(self.qblox, f'{m}'.lower())
            
            # Steal code from Qblox official tutoirals.
            if self[f'{m}/type'] == 'QCM-RF':
                for sequencer in this_module.sequencers:
                    for out in range(0, 4):
                        if hasattr(sequencer, "channel_map_path{}_out{}_en".format(out%2, out)):
                            sequencer.set("channel_map_path{}_out{}_en".format(out%2, out), False)
                    
            elif self[f'{m}/type'] == 'QRM-RF':
                for sequencer in this_module.sequencers:
                    for out in range(0, 2):
                        if hasattr(sequencer, "channel_map_path{}_out{}_en".format(out%2, out)):
                            sequencer.set("channel_map_path{}_out{}_en".format(out%2, out), False)
                
            else:
                raise ValueError(f'The type of {m} is invalid.')

















