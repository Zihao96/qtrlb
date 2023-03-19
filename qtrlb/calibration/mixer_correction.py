import os
import json
import numpy as np




class MixerCorrection:
    """ A simple class for mixer correction.
    
        Attributes:
            cfg: A MetaManager
            qudit: 'Q2', 'Q3', 'R4', 'R1'
            subspace: '01', '12'. Only work for qubit, not resonator.
            amp: Float number between 0 and 1.
            waveform_length: Integer between [4,16384]. Don't change.
    """
    def __init__(self, 
                 cfg, 
                 qudit: str,
                 subspace: str = '01',
                 amp: float = 0.8,
                 waveform_length: int = 120):
        self.cfg = cfg
        self.qudit = qudit
        self.subspace = subspace
        self.amp = amp
        self.waveform_length = waveform_length
        
        # There are pointer to actual object in qblox driver.
        self.module = cfg.DAC.module[qudit]
        self.sequencer = cfg.DAC.sequencer[qudit]
        
        
    def run(self):
        """
        Make sequence and start sequencer.
        """
        self.make_sequence()
        self.sequencer.arm_sequencer()
        self.cfg.DAC.qblox.start_sequencer()
        
        
    def make_sequence(self):
        """
        Set up sequence and configure the instrument.
        
        Reference: 
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/cont_wave_mode.html
        """
        waveform = {'waveform':{'data':np.ones(self.waveform_length, dtype=float).tolist(), 'index':0}}
        sequence = {'waveforms':waveform, 'weights':{}, 'acquisitions':{}, 'program':'stop'}

        file_path = os.path.join(self.cfg.working_dir, 'Jsons', 'SSB_Calibration.json')
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(sequence, file, indent=4)


        self.cfg.DAC.qblox.reset()
        self.sequencer.sequence(file_path)
        self.sequencer.sync_en(True)
        self.sequencer.mod_en_awg(True)
        self.sequencer.marker_ovr_en(True)   
        self.sequencer.marker_ovr_value(15)
        self.sequencer.gain_awg_path0(self.amp)
        self.sequencer.gain_awg_path1(self.amp)
        self.sequencer.cont_mode_en_awg_path0(True)
        self.sequencer.cont_mode_en_awg_path1(True)
        self.sequencer.cont_mode_waveform_idx_awg_path0(0)  
        self.sequencer.cont_mode_waveform_idx_awg_path1(0)


        if self.qudit.startswith('Q'):
            # Because the name of attribute depends on which output port.
            self.out = self.cfg[f'variables.{self.qudit}/out']
            attr = getattr(self.module, f'out{self.out}_lo_freq')
            attr(self.cfg[f'variables.{self.qudit}/qubit_LO'])
            attr = getattr(self.sequencer, f'channel_map_path0_out{self.out * 2}_en')
            attr(True)
            attr = getattr(self.sequencer, f'channel_map_path1_out{self.out * 2 + 1}_en')
            attr(True)
            self.sequencer.nco_freq(self.cfg[f'variables.{self.qudit}/{self.subspace}/mod_freq'])
        elif self.qudit.startswith('R'):
            self.out = 0
            self.module.out0_in0_lo_freq(self.cfg[f'variables.{self.qudit}/resonator_LO'])
            self.sequencer.channel_map_path0_out0_en(True)
            self.sequencer.channel_map_path1_out1_en(True)
            self.sequencer.nco_freq(self.cfg[f'variables.{self.qudit}/mod_freq'])
            
            
    def set_offset0(self, offset0):
        """
        offset0 should be float number between [-84, 73].
        """
        attr = getattr(self.module, f'out{self.out}_offset_path0')
        attr(offset0)


    def set_offset1(self, offset1):
        """
        offset1 should be float number between [-84, 73].
        """
        attr = getattr(self.module, f'out{self.out}_offset_path1')
        attr(offset1)


    def set_gain_ratio(self, gain_ratio):
        """
        gain_ratio should be float number between [0.5, 2]. It usually takes [0.9, 1.1].
        """
        self.sequencer.mixer_corr_gain_ratio(gain_ratio)
        self.sequencer.arm_sequencer()
        self.cfg.DAC.qblox.start_sequencer()


    def set_phase_offset(self, phase_offset):
        """
        phase_offset should be float number between [-45, 45].
        """
        self.sequencer.mixer_corr_phase_offset_degree(phase_offset)
        self.sequencer.arm_sequencer()
        self.cfg.DAC.qblox.start_sequencer()