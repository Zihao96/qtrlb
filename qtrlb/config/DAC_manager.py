import os
import time
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager
from qblox_instruments import Cluster


class DACManager(Config):
    """ This is a thin wrapper over the Config class to help with DAC management.
        The load() method will be called once in its __init__.
    
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
                         suffix='DAC',
                         varman=varman)
        
        # Call parent load once to get name and address in DAC.yaml.
        super().load()

        # Connect to instrument.
        Cluster.close_all()
        self.test_mode = test_mode
        if self.test_mode:
            dummy_cfg = {2:'Cluster QCM-RF', 4:'Cluster QCM-RF', 6:'Cluster QCM-RF', 8:'Cluster QRM-RF'}
            self.qblox = Cluster(name='cluster', dummy_cfg=dummy_cfg)
        else:
            self.qblox = Cluster(self['name'], self['address']) 

        self.qblox.reset()
        self.load()
    
    
    def load(self):
        """
        Run the parent load and add new dictionary attributes.
        These attributes contain pointer to real object in instrument driver.
        """
        super().load()
        
        self.module = {}
        self.sequencer = {}
        for tone in self.varman['tones']:
            self.module[tone] = getattr(self.qblox, 'module{}'.format(self.varman[f'{tone}/mod']))
            self.sequencer[tone] = getattr(self.module[tone], 'sequencer{}'.format(self.varman[f'{tone}/seq']))
             
        
    def implement_parameters(self, tones: list, jsons_path: str):
        """
        Implement the setting/parameters onto Qblox.
        This function should be called after we know which specific tones will be used.
        The tones should be list of string like: ['Q3/01', 'Q3/12', 'Q3/23', 'Q4/01', 'Q4/12', 'R3', 'R4'].
        """
        self.qblox.reset()
        self.disconnect_existed_map()
        self.disable_all_lo()
        
        for tone in tones:
            tone_ = tone.replace('/', '_')
            mod = self.varman[f'{tone}/mod']  # A string of integer index for convenience.
            out = self.varman[f'{tone}/out']
            seq = self.varman[f'{tone}/seq']

            # Implement common parameters.
            for key, value in self[f'Module{mod}'].items():
                if key.startswith(('out', 'in', 'scope')): getattr(self.module[tone], key)(value)

            # Implement QCM-RF specific parameters.
            if tone.startswith('Q'):
                getattr(self.module[tone], f'out{out}_lo_en')(True)
                time.sleep(0.005)  # This 5ms sleep is important to make LO work correctly. 1ms doesn't work.
                getattr(self.module[tone], f'out{out}_lo_freq')(self.varman[f'lo_freq/M{mod}O{out}'])
                self.sequencer[tone].sync_en(True)
                self.sequencer[tone].mod_en_awg(True)
                getattr(self.sequencer[tone], f'channel_map_path0_out{out*2}_en')(True)
                getattr(self.sequencer[tone], f'channel_map_path1_out{out*2+1}_en')(True)
                
            # Implement QRM-RF specific parameters.
            elif tone.startswith('R'):
                self.module[tone].out0_in0_lo_en(True)
                time.sleep(0.005)  # This 5ms sleep is important to make LO work correctly. 1ms doesn't work.
                self.module[tone].out0_in0_lo_freq(self.varman[f'lo_freq/M{mod}O{out}'])
                self.module[tone].scope_acq_sequencer_select(seq)  # Last sequencer to triger acquire.
                self.sequencer[tone].sync_en(True)
                self.sequencer[tone].mod_en_awg(True)
                self.sequencer[tone].demod_en_acq(True)
                self.sequencer[tone].integration_length_acq(round(self.varman['common/integration_length'] * 1e9))
                self.sequencer[tone].nco_prop_delay_comp_en(self.varman['common/nco_delay_comp'])
                self.sequencer[tone].channel_map_path0_out0_en(True)
                self.sequencer[tone].channel_map_path1_out1_en(True)
                  
            # Correct sideband tone of mixer. Nulling LO tone was applied in common parameters.
            self.sequencer[tone].mixer_corr_gain_ratio(
                self[f'Module{mod}/Sequencer{seq}/mixer_corr_gain_ratio']
            )           
            self.sequencer[tone].mixer_corr_phase_offset_degree(
                self[f'Module{mod}/Sequencer{seq}/mixer_corr_phase_offset_degree']
            )
            
            # Upload sequence json file to instrument.
            file_path = os.path.join(jsons_path, f'{tone_}_sequence.json')
            self.sequencer[tone].sequence(file_path)
    
    
    def disconnect_existed_map(self):
        """
        Disconnect all existed maps between two output paths of each output port 
        and two output paths of each sequencer.
        """
        for module in self.qblox.modules:
            if not (module.present() and module.is_rf_type): continue

            if module.is_qcm_type:
                for sequencer in module.sequencers:
                    for out in range(0, 4):
                        if hasattr(sequencer, "channel_map_path{}_out{}_en".format(out%2, out)):
                            sequencer.set("channel_map_path{}_out{}_en".format(out%2, out), False)

            elif module.is_qrm_type:
                for sequencer in module.sequencers:
                    for out in range(0, 2):
                        if hasattr(sequencer, "channel_map_path{}_out{}_en".format(out%2, out)):
                            sequencer.set("channel_map_path{}_out{}_en".format(out%2, out), False)

            else:
                print(f'Failed to disconnect channel map for module type {module.module_type}')
            

    def disable_all_lo(self):
        """
        Disable all the LO so that nothing will come out of output port when we don't use them.
        It's really because all LO is enabled by default.
        """
        for module in self.qblox.modules:
            if not (module.present() and module.is_rf_type): continue

            if module.is_qcm_type:
                module.out0_lo_en(False)
                module.out1_lo_en(False)
            elif module.is_qrm_type:
                module.out0_in0_lo_en(False)
            else:
                print(f'Failed to disable LO for module type {module.module_type}')


    def start_sequencer(self, tones: list, measurement: dict, keep_raw: bool = False, heralding_enable: bool = False):
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
        for tone in tones:
            self.sequencer[tone].arm_sequencer()
            
        # Really start sequencer.
        self.qblox.start_sequencer()  

        for r in tones:
            # Only loop over resonator.
            if not r.startswith('R'): continue

            timeout = self['Module{}/acquisition_timeout'.format(self.varman[f'{r}/mod'])]
            seq_idx = int(self.varman[f'{r}/seq'])
           
            # Wait the timeout in minutes and ask whether the acquisition finish on that sequencer. Raise error if not.
            self.module[r].get_acquisition_state(seq_idx, timeout)  

            # Store the raw (scope) data from buffer of FPGA to RAM of instrument.
            if keep_raw: 
                self.module[r].store_scope_acquisition(seq_idx, 'readout')
                if heralding_enable: self.module[r].store_scope_acquisition(seq_idx, 'heralding')
            
            # Retrive the heterodyned result (binned data) back to python in Host PC.
            data = self.module[r].get_acquisitions(seq_idx)
            
            # Clear the memory of instrument. 
            # It's necessary otherwise the acquisition result will accumulate and be averaged.
            self.module[r].delete_acquisition_data(seq_idx, 'readout')
            if heralding_enable: self.module[r].delete_acquisition_data(seq_idx, 'heralding')
            
            # Append list of each repetition into measurement dictionary.
            measurement[r]['Heterodyned_readout'][0].append(data['readout']['acquisition']['bins']['integration']['path0']) 
            measurement[r]['Heterodyned_readout'][1].append(data['readout']['acquisition']['bins']['integration']['path1'])
            if heralding_enable:
                measurement[r]['Heterodyned_heralding'][0].append(data['heralding']['acquisition']['bins']['integration']['path0']) 
                measurement[r]['Heterodyned_heralding'][1].append(data['heralding']['acquisition']['bins']['integration']['path1'])

            if keep_raw:
                measurement[r]['raw_readout'][0].append(data['readout']['acquisition']['scope']['path0']['data']) 
                measurement[r]['raw_readout'][1].append(data['readout']['acquisition']['scope']['path1']['data'])
                if heralding_enable:
                    measurement[r]['raw_heralding'][0].append(data['heralding']['acquisition']['scope']['path0']['data']) 
                    measurement[r]['raw_heralding'][1].append(data['heralding']['acquisition']['scope']['path1']['data'])

        # In case of the sequencers don't stop correctly.
        # Do not call qblox.reset() here since it will make debugging difficult.
        self.qblox.stop_sequencer()

