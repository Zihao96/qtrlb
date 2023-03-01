import os
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
        
        # Connect to instrument. Hardcode name and IP address to accelerate load()
        Cluster.close_all()
        self.qblox = Cluster('cluster', '192.168.0.2')  # TODO: Change it back when finish.
        # dummy_cfg = {2:'Cluster QCM-RF', 4:'Cluster QCM-RF', 6:'Cluster QCM-RF', 8:'Cluster QRM-RF'}
        # self.qblox = Cluster(name='cluster', dummy_cfg=dummy_cfg)
        self.qblox.reset()
        
        self.load()
    
    
    def load(self):
        """
        Run the parent load and add necessary new item in config_dict.
        """
        super().load()
        
        modules_list = [key for key in self.keys() if key.startswith('Module')]
        self.set('modules', modules_list, which='dict')  # Keep the new key start with lowercase!
        
        # These dictionary contain pointer to real object in instrument driver.
        self.module = {}
        self.sequencer = {}
        for qudit in self.varman['qubits'] + self.varman['resonators']:
            self.module[qudit] = getattr(self.qblox, 'module{}'.format(self.varman[f'{qudit}/module']))
            self.sequencer[qudit] = getattr(self.module[qudit], 'sequencer{}'.format(self.varman[f'{qudit}/sequencer']))
             
        
    def implement_parameters(self, qubits: list, resonators: list, jsons_path: str):
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
        for q in qubits:
            qubit_module = self.varman[f'{q}/module']  # Just an interger. It's for convenience.
            
            for attribute in self[f'Module{qubit_module}'].keys():
                if attribute.startswith('out') or attribute.startswith('in'):
                    attr = getattr(self.module[q], attribute)
                    attr(self[f'Module{qubit_module}/{attribute}'])
            
            attr = getattr(self.module[q], 'out{}_lo_freq'.format(self.varman[f'{q}/out']))
            attr(self.varman[f'{q}/qubit_LO'])        
            self.sequencer[q].sync_en(True)
            self.sequencer[q].mod_en_awg(True)
            attr = getattr(self.sequencer[q], 'channel_map_path0_out{}_en'.format(self.varman[f'{q}/out'] * 2))
            attr(True)
            attr = getattr(self.sequencer[q], 'channel_map_path1_out{}_en'.format(self.varman[f'{q}/out'] * 2 + 1))
            attr(True)
            
            # self.sequencer[q].gain_awg_path0(self.varman[f'{q}/{subspace[i]}/amp_rabi'])
            # self.sequencer[q].gain_awg_path1(self.varman[f'{q}/{subspace[i]}/amp_rabi'])
            # self.sequencer[q].nco_freq(self.varman[f'{q}/{subspace[i]}/mod_freq']) 
            self.sequencer[q].mixer_corr_gain_ratio(self[f'Module{qubit_module}/mixer_corr_gain_ratio'])           
            self.sequencer[q].mixer_corr_phase_offset_degree(self[f'Module{qubit_module}/mixer_corr_phase_offset_degree'])
            file_path = os.path.join(jsons_path, f'{q}_sequence.json')
            self.sequencer[q].sequence(file_path)

        
        for r in resonators:
            resonator_module = self.varman[f'{r}/module']
            
            for attribute in self[f'Module{resonator_module}'].keys():
                if attribute.startswith('out') or attribute.startswith('in') or attribute.startswith('scope'):
                    attr = getattr(self.module[r], attribute)
                    attr(self[f'Module{resonator_module}/{attribute}'])
            
            self.module[r].out0_in0_lo_freq(self.varman[f'{r}/resonator_LO'])        
            self.module[r].scope_acq_sequencer_select(self.varman[f'{r}/sequencer'])  # Last sequencer to triger acquire.
            self.sequencer[r].sync_en(True)
            self.sequencer[r].mod_en_awg(True)
            self.sequencer[r].demod_en_acq(True)
            self.sequencer[r].integration_length_acq(round(self.varman['common/integration_length'] * 1e9))
            self.sequencer[r].channel_map_path0_out0_en(True)
            self.sequencer[r].channel_map_path1_out1_en(True)

            # self.sequencer[r].gain_awg_path0(self.varman[f'{r}/amp'])
            # self.sequencer[r].gain_awg_path1(self.varman[f'{r}/amp'])
            # self.sequencer[r].nco_freq(self.varman[f'{r}/mod_freq'])                    
            self.sequencer[r].mixer_corr_gain_ratio(self[f'Module{resonator_module}/mixer_corr_gain_ratio'])
            self.sequencer[r].mixer_corr_phase_offset_degree(self[f'Module{resonator_module}/mixer_corr_phase_offset_degree'])
            file_path = os.path.join(jsons_path, f'{r}_sequence.json')
            self.sequencer[r].sequence(file_path)
            
        # Sorry, this is just temporary code. It's tricky to set those freq/amp based on which qubit. --Zihao (01/30/2023)
    
    
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


    def start_sequencer(self, qubits: list, resonators: list, measurement: dict, keep_raw: bool = False):
        """
        Ask the instrument to start sequencer.
        Then store the Heterodyned result into measurement.
        
        Reference about data structure:
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/cluster.html#qblox_instruments.native.Cluster.get_acquisitions
        
        Note from Zihao(02/15/2023):
        We need to call delete_scope_acquisition everytime.
        Otherwise the binned result will accumulate and be averaged automatically for next repetition.
        Within one repetition (one round), we can only keep raw data from last bin (last acquire instruction).
        Which means for Scan, only the raw trace belong to last point in x_points will be stored.
        So it's barely useful, but I still leave the interface here.
        """
        # Arm sequencer first. It's necessary. Only armed sequencer will be started next.
        for qudit in qubits + resonators:
            self.sequencer[qudit].arm_sequencer()
            
        # Really start sequencer.
        self.qblox.start_sequencer()  

        for r in resonators:
            timeout = self['Module{}/acquisition_timeout'.format(self.varman[f'{r}/module'])]
            seq_num = self.varman[f'{r}/sequencer']
           
            # Wait the timeout in minutes and ask whether the acquisition finish on that sequencer. Raise error if not.
            self.module[r].get_acquisition_state(seq_num, timeout)  

            # Store the raw (scope) data from buffer of FPGA to RAM of instrument.
            if keep_raw: 
                self.module[r].store_scope_acquisition(seq_num, 'readout')
                self.module[r].store_scope_acquisition(seq_num, 'heralding')
            
            # Retrive the heterodyned result (binned data) back to python in Host PC.
            data = self.module[r].get_acquisitions(seq_num)
            
            # Clear the memory of instrument. 
            # It's necessary otherwise the acquisition result will accumulate and be averaged.
            self.module[r].delete_acquisition_data(seq_num, 'readout')
            self.module[r].delete_acquisition_data(seq_num, 'heralding')
            
            # Append list of each repetition into measurement dictionary.
            measurement[r]['Heterodyned_readout'][0].append(data['readout']['acquisition']['bins']['integration']['path0']) 
            measurement[r]['Heterodyned_readout'][1].append(data['readout']['acquisition']['bins']['integration']['path1'])
            measurement[r]['Heterodyned_heralding'][0].append(data['heralding']['acquisition']['bins']['integration']['path0']) 
            measurement[r]['Heterodyned_heralding'][1].append(data['heralding']['acquisition']['bins']['integration']['path1'])

            if keep_raw:
                measurement[r]['raw_readout'][0].append(data['readout']['acquisition']['scope']['path0']['data']) 
                measurement[r]['raw_readout'][1].append(data['readout']['acquisition']['scope']['path1']['data'])
                measurement[r]['raw_heralding'][0].append(data['heralding']['acquisition']['scope']['path0']['data']) 
                measurement[r]['raw_heralding'][1].append(data['heralding']['acquisition']['scope']['path1']['data'])

