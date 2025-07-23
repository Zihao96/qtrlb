import os
import time
import json
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
            dummy_cfg = {2:'Cluster QCM-RF', 6:'Cluster QCM-RF', 10:'Cluster QCM-RF', 14:'Cluster QRM-RF'}
            self.qblox = Cluster(name='cluster', dummy_cfg=dummy_cfg)
        else:
            self.qblox = Cluster(self['name'], self['address']) 
            self.reset()

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


    def reset(self):
        """
        Reset the instrument to default state.
        We will also fix fan speed if needed.
        """
        self.qblox.reset()
        self.set_automated_control(self['automated_control'])
        if not self['automated_control']: self.set_fan_speed(self['fan_speed'])

        
    def implement_parameters(self, tones: list, jsons_path: str):
        """
        Implement the setting/parameters onto Qblox.
        This function should be called after we know which specific tones will be used.
        The tones should be list of string like: ['Q3/01', 'Q3/12', 'Q3/23', 'Q4/01', 'Q4/12', 'R3', 'R4'].
        """
        self.reset()
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
                getattr(self.sequencer[tone], f'connect_out{out}')('IQ')
                
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
                self.sequencer[tone].connect_out0('IQ')
                self.sequencer[tone].connect_acq('in0')
                  
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
    

    def set_automated_control(self, on: bool) -> None:
        """
        Set the automated control on/off of Qblox Cluster.
        """
        if not isinstance(on, bool): raise TypeError(f'DACManager: on {on} is not boolean variables.')
        return self.qblox._write(f'CTRL:ENA {int(on)}')
    

    def get_fan_speed(self) -> int:
        """
        Get the fan speed of Qblox Cluster in RPM.
        """
        return int(self.qblox._read('STATus:QUEStionable:FANS:BP0:SPEed?'))


    def set_fan_speed(self, RPM: int) -> None:
        """
        Set the fan speed of Qblox Cluster in RPM.
        """
        if not 0 < int(RPM) < 5400: raise ValueError(f'DACManager: RPM {RPM} is not in range (0, 5400).')
        return self.qblox._write(f'STATus:QUEStionable:FANS:BP0:SPEed {int(RPM)}')
    

    def set_led_brightness(self, brightness: str) -> None:
        if brightness not in ('high', 'low', 'off', 'medium'): 
            raise ValueError(f'DACManager: brightness must be in ("high", "low", "off", "medium").')
        return self.qblox.led_brightness(brightness)

    
    def disconnect_existed_map(self):
        """
        Disconnect all existed maps between two output paths of each output port 
        and two output paths of each sequencer.
        """
        for module in self.qblox.modules:
            if not (module.present() and module.is_rf_type): continue

            if module.is_qcm_type:
                module.disconnect_outputs()
            elif module.is_qrm_type:
                module.disconnect_outputs()
                module.disconnect_inputs()
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


    def start_sequencer(self, tones: list, readout_tones: list, measurement: dict, jsons_path: str, 
                        keep_raw: bool = False) -> None:
        """
        Start sequencer, run the sequence program, and store the acquisition results.
        Please refer to the Qblox official documentation of the get_acquisitions() method \
            in Sequencer API for the data structure.
        
        Each time we start sequencer and do acquisition, we can only keep raw (scope) data \
            associated with the last bin (the last 'acquire' instruction), because the \
            scope data in the buffer is always overwritten unless we store it to FPGA. \
            For 1D scan, it means we can only keep the raw data of the last point in x_points.
            It's occasionally useful.
        """
        # Arm sequencer first. It's necessary. Only armed sequencer will be started next.
        for tone in tones:
            self.sequencer[tone].arm_sequencer()
            
        # Really start sequencer.
        self.qblox.start_sequencer()  

        for rt in readout_tones:
            rr, subtone = rt.split('/')
            timeout = self['Module{}/acquisition_timeout'.format(self.varman[f'{rt}/mod'])]

            # We load the sequence dictionary to get all keys in the acquisition.
            with open(os.path.join(jsons_path, f'{rr}_{subtone}_sequence.json'), 'r', encoding='utf-8') as file:
                acq_keys = json.load(file)['acquisitions'].keys()

            # Block the program until acquisition finish. Raise error when blocking time is longer than timeout (minutes).
            # The timeout must be a non-zero positive integer.
            self.sequencer[rt].get_acquisition_status(timeout)  

            # Store the raw (scope) data from the buffer of the ADC to the larger RAM on the FPGA.
            if keep_raw: 
                for key in acq_keys:
                    self.sequencer[rt].store_scope_acquisition(key)
            
            # Retrive the acquisition data from the FPGA to the host PC.
            data = self.sequencer[rt].get_acquisitions()
            
            for key in acq_keys:
                # Add acquisition data into measurement dictionary.
                measurement[rr][subtone][f'Heterodyned_{key}'][0].append(data[key]['acquisition']['bins']['integration']['path0']) 
                measurement[rr][subtone][f'Heterodyned_{key}'][1].append(data[key]['acquisition']['bins']['integration']['path1'])

                if keep_raw:
                    measurement[rr][subtone][f'raw_{key}'][0].append(data[key]['acquisition']['scope']['path0']['data']) 
                    measurement[rr][subtone][f'raw_{key}'][1].append(data[key]['acquisition']['scope']['path1']['data'])

                # Clear both the stored scope acquisition and the bin acquisition on the FPGA.
                # It's necessary otherwise the acquisition result will accumulate and be averaged.
                # The scope data in the buffer of the ADC is always automatically overwritten.
                self.sequencer[rt].delete_acquisition_data(all=True)

        # In case that the sequencers don't stop correctly.
        self.qblox.stop_sequencer()

