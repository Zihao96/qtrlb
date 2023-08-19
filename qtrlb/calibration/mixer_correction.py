import os
import json
import time
import numpy as np
import ipywidgets as ipyw
import qtrlb.utils.units as u
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qtrlb.config.config import MetaManager
from qtrlb.utils.N9010A import N9010A




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
                 cfg: MetaManager, 
                 qudit: str,
                 subspace: str = '01',
                 amp: float = 0.1,
                 waveform_length: int = 40):
        self.cfg = cfg
        self.qudit = qudit
        self.subspace = subspace
        self.amp = amp
        self.waveform_length = waveform_length
        self.tone = f'{qudit}/{subspace}' if qudit.startswith('Q') else qudit
        
        # There are pointer to actual object in qblox driver.
        self.module = cfg.DAC.module[self.qudit]
        self.sequencer = cfg.DAC.sequencer[self.tone]

        self.module_idx = self.module._slot_idx
        self.sequencer_idx = self.sequencer._seq_idx
        
        
    def run(self):
        """
        Make sequence, start sequencer and create interactive ipywidget.
        """
        self.make_sequence()
        self.create_ipywidget()
        
        
    def make_sequence(self):
        """
        Set up sequence and configure the instrument.
        
        Reference: 
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/cont_wave_mode.html
        """
        sequence = {'waveforms': {'waveform':{'data': np.ones(self.waveform_length, dtype=float).tolist(), 
                                              'index': 0}}, 
                    'weights': {}, 
                    'acquisitions': {}, 
                    'program': f"""
                        wait_sync   4

                        loop:       play    0,0,{self.waveform_length}
                                    jmp     @loop
                        """
                    }

        file_path = os.path.join(self.cfg.working_dir, 'Jsons', 'SSB_Calibration.json')
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(sequence, file, indent=4)

        self.cfg.DAC.qblox.reset()
        self.cfg.DAC.disconnect_existed_map()
        self.cfg.DAC.disable_all_lo()
        self.sequencer.sequence(file_path)
        self.sequencer.sync_en(True)
        self.sequencer.mod_en_awg(True)
        self.sequencer.marker_ovr_en(True)   
        self.sequencer.marker_ovr_value(15)
        self.sequencer.gain_awg_path0(self.amp)
        self.sequencer.gain_awg_path1(self.amp)

        if self.qudit.startswith('Q'):
            # Because the name of attribute depends on which output port.
            self.out = self.cfg[f'variables.{self.qudit}/out']
            self.att = self.cfg[f'DAC.Module{self.module_idx}/out{self.out}_att']
            getattr(self.module, f'out{self.out}_lo_en')(True)
            time.sleep(0.005)  # This sleep is important to make LO work correctly. 1 ms doesn't work.
            getattr(self.module, f'out{self.out}_lo_freq')(self.cfg[f'variables.{self.qudit}/qubit_LO'])
            getattr(self.module, f'out{self.out}_att')(self.att)
            getattr(self.sequencer, f'channel_map_path0_out{self.out * 2}_en')(True)
            getattr(self.sequencer, f'channel_map_path1_out{self.out * 2 + 1}_en')(True)
            self.sequencer.nco_freq(self.cfg[f'variables.{self.tone}/mod_freq'])

        elif self.qudit.startswith('R'):
            self.out = 0
            self.att = self.cfg[f'DAC.Module{self.module_idx}/out0_att']
            self.module.out0_in0_lo_en(True)
            time.sleep(0.005)  # This sleep is important to make LO work correctly. 1 ms doesn't work.
            self.module.out0_in0_lo_freq(self.cfg[f'variables.{self.qudit}/resonator_LO'])
            self.module.out0_att(self.att)
            self.sequencer.channel_map_path0_out0_en(True)
            self.sequencer.channel_map_path1_out1_en(True)
            self.sequencer.nco_freq(self.cfg[f'variables.{self.qudit}/mod_freq'])
            self.sequencer.nco_prop_delay_comp_en(True)


    def create_ipywidget(self, 
                         offset0_min: float = -84.0,
                         offset0_max: float = +73.0,
                         offset1_min: float = -84.0,
                         offset1_max: float = +73.0,
                         widget_width: str = '600px', ) -> None:
        """
        Start sequencer once and create ipywidget.
        Allow user to change min/max of the FloatSlider for DC offset. 
        """
        
        layout = ipyw.Layout(width=widget_width) 
        self.sequencer.arm_sequencer()
        self.cfg.DAC.qblox.start_sequencer()
        
        ipyw.interact(
            self.set_offset0, 
            offset0=ipyw.FloatSlider(
                value=self.cfg[f'DAC.Module{self.module_idx}/out{self.out}_offset_path0'], 
                min=offset0_min, max=offset0_max, step=0.001, layout=layout
            )
        )
        ipyw.interact(
            self.set_offset1, 
            offset1=ipyw.FloatSlider(
                value=self.cfg[f'DAC.Module{self.module_idx}/out{self.out}_offset_path1'], 
                min=offset1_min, max=offset1_max, step=0.001, layout=layout
            )
        )
        ipyw.interact(
            self.set_gain_ratio, 
            gain_ratio=ipyw.FloatSlider(
                value=self.cfg[f'DAC.Module{self.module_idx}/Sequencer{self.sequencer_idx}/mixer_corr_gain_ratio'], 
                min=0.7, max=1.3, step=0.001, layout=layout
            )
        )
        ipyw.interact(
            self.set_phase_offset, 
            phase_offset=ipyw.FloatSlider(
                value=self.cfg[f'DAC.Module{self.module_idx}/Sequencer{self.sequencer_idx}/mixer_corr_phase_offset_degree'], 
                min=-45.0, max=45.0, step=0.001, layout=layout
            )   
        )


    def stop(self, save_cfg: bool = False):
        """
        Stop sequencer and store all the current value into cfg if asked.
        """
        self.cfg.DAC.qblox.stop_sequencer()
        if save_cfg:
            offset0 = getattr(self.module, f'out{self.out}_offset_path0')()
            offset1 = getattr(self.module, f'out{self.out}_offset_path1')()
            gain_ratio = self.sequencer.mixer_corr_gain_ratio()
            phase_offset = self.sequencer.mixer_corr_phase_offset_degree()

            self.cfg[f'DAC.Module{self.module_idx}/out{self.out}_offset_path0'] = offset0
            self.cfg[f'DAC.Module{self.module_idx}/out{self.out}_offset_path1'] = offset1
            self.cfg[f'DAC.Module{self.module_idx}/Sequencer{self.sequencer_idx}/mixer_corr_gain_ratio'] = gain_ratio
            self.cfg[f'DAC.Module{self.module_idx}/Sequencer{self.sequencer_idx}/mixer_corr_phase_offset_degree'] = phase_offset
            self.cfg.save()
            self.cfg.load()
            
            
    def set_offset0(self, offset0):
        """
        offset0 should be float number between [-84, 73].
        """
        getattr(self.module, f'out{self.out}_offset_path0')(offset0)


    def set_offset1(self, offset1):
        """
        offset1 should be float number between [-84, 73].
        """
        getattr(self.module, f'out{self.out}_offset_path1')(offset1)


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




class MixerAutoCorrection(MixerCorrection):
    """ Automate mixer correction by using Spectrum Analyzer N9010A.
        The sb in code refer to the mirrored sideband in the signal.
        It's not the one we want to keep.
    
        Attributes:
            sa: A spectrum analyer object.
            cfg: A MetaManager
            qudit: 'Q2', 'Q3', 'R4', 'R1'
            subspace: '01', '12'. Only work for qubit, not resonator.
            amp: Float number between 0 and 1.
            waveform_length: Integer between [4,16384]. Don't change.
    """
    def __init__(self, 
                 sa: N9010A,
                 cfg: MetaManager, 
                 qudit: str,
                 subspace: str = '01',
                 amp: float = 0.1,
                 waveform_length: int = 40):
        
        super().__init__(cfg, qudit, subspace, amp, waveform_length)
        self.sa = sa

        # Get the exact frequency for convenience.
        lo_key = 'qubit_lo' if self.qudit.startswith('Q') else 'resonator_lo'
        self.main_freq = self.cfg[f'variables.{self.tone}/freq']
        self.lo_freq = self.cfg[f'variables.{self.qudit}/{lo_key}']
        self.sb_freq = self.lo_freq * 2 - self.main_freq
        self.mod_freq = self.main_freq - self.lo_freq


    def run(self,
            which: str = 'both',
            save_cfg: bool = False,
            method: str = 'Powell',
            readout_delay_time: float = 0.02,
            readout_avg_num: int = 5,
            lo_maxiter: int = 3,
            sb_maxiter: int = 2):
        """
        Run optimization on specified tones.

        Parameters:
        readout_delay_time: The time 
        """
        
        self.minimize_method = method
        self.readout_delay_time = readout_delay_time
        self.readout_avg_num = readout_avg_num
        self.lo_maxiter = lo_maxiter
        self.sb_maxiter = sb_maxiter

        # Start signal output and load current parameter.
        self.make_sequence()
        self.set_offset0(
            self.cfg[f'DAC.Module{self.module_idx}/out{self.out}_offset_path0']
        )
        self.set_offset1(
            self.cfg[f'DAC.Module{self.module_idx}/out{self.out}_offset_path1']
        )
        self.set_gain_ratio(
            self.cfg[f'DAC.Module{self.module_idx}/Sequencer{self.sequencer_idx}/mixer_corr_gain_ratio']
        )
        self.set_phase_offset(
            self.cfg[f'DAC.Module{self.module_idx}/Sequencer{self.sequencer_idx}/mixer_corr_phase_offset_degree']
        )

        # Start optimization.
        self.set_sa()
        self.old_spectrum = self.sa.data

        if which.lower() == 'both':
            self.minimize_lo()
            self.minimize_sb()
        elif which.lower() == 'lo':
            self.minimize_lo()
        elif which.lower() == 'sb':
            self.minimize_sb()
        else:
            raise ValueError(f'MixerAutoCorrection: The "which" argument can only take "both", "lo", "sb".')

        self.new_spectrum = self.sa.data
        self.stop(save_cfg)
        self.plot()


    def set_sa(self):
        """
        Set up instrument before doing optimization.
        """
        # A wide span with all three peaks on screen.
        self.sa.set('freq_center', self.lo_freq)
        self.sa.set('freq_span', 2.4 * abs(self.mod_freq))
        self.sa.set('res_bw', abs(self.mod_freq) * 1e-4)
        self.sa.set('vid_bw_auto')
        self.sa.set('ref_level', '15')

        # Set three markers on three peaks.
        self.sa.set_marker(1, 'ON')
        self.sa.set_marker(2, 'ON')
        self.sa.set_marker(3, 'ON')
        self.sa.set_marker_center(1)
        self.sa.set_marker_center(2)
        self.sa.set_marker_center(3)

        # Maddy leave the 0.1 second delay here and probably has her reason.
        peak_left, peak_right = 3, 1 if self.mod_freq > 0 else 1, 3
        time.sleep(0.1)
        self.sa.set_marker('peak_left', peak_left)
        time.sleep(0.1)
        self.sa.set_marker('peak_right', peak_right)

        # Set up the proper reference level
        peak_highest = np.max([float(self.sa.get_marker('y', i+1)) for i in range(3)])
        ref_level = np.round(peak_highest) + 5
        self.sa.set('ref_level', ref_level)


    def minimize_lo(self):
        """
        Minimize the LO tone using scipy.optimize.minimize.
        Here we zoom into the LO tone such that there is only one wide peak on screen.
        """
        self.sa.set('freq_center', self.lo_freq)
        self.sa.set('freq_span', '2MHz')
        self.sa.set('res_bw', '10kHz')
        result = minimize(self.loss_func_lo, x0=(0,0), method=self.minimize_method, 
                          bounds=((-84,73), (-84,73)), options = {'maxiter' : self.lo_maxiter})
        self.set_offset0(result.x[0])
        self.set_offset1(result.x[1])


    def loss_func_lo(self, x) -> float:
        """
        The function we pass into minimize for reduce LO tone.
        Here x[0] will be offset0, x[1] will be offset1.
        Here we read marker several times and take average to reduce fluctuation.
        """
        self.set_offset0(x[0])
        self.set_offset1(x[1])
        time.sleep(self.readout_delay_time)

        return np.mean([float(self.sa.get_marker('y', 2)) for _ in range(self.readout_avg_num)])
    

    def minimize_sb(self):
        """
        Minimize the SB tone using scipy.optimize.minimize.
        Here we zoom into the SB tone such that there is only one wide peak on screen.
        We amplify the gain ratio (0.5-2.0) 100 times to avoid vanishing/exploding gradients.
        """
        self.sa.set('freq_center', self.sb_freq)
        self.sa.set('freq_span', '2MHz')
        self.sa.set('res_bw', '10kHz')
        result = minimize(self.loss_func_sb, x0=(100, 0), method=self.minimize_method, 
                          bounds=((50,200), (-45,45)), options={'maxiter': self.sb_maxiter})
        self.set_gain_ratio(result.x[0]/100)
        self.set_phase_offset(result.x[1])


    def loss_func_sb(self, x) -> float:
        """
        The function we pass into minimize for reduce SB tone.
        Here x[0] will be offset0, x[1] will be offset1.
        Here we read marker several times and take average to reduce fluctuation.
        """
        self.set_gain_ratio(x[0]/100)
        self.set_phase_offset(x[1])
        time.sleep(self.readout_delay_time)

        return np.mean([float(self.sa.get_marker('y', 3)) for _ in range(self.readout_avg_num)])
    

    def plot(self):
        """
        Plot the spectrum before/after mixer correction for user to compare them.
        """
        fig, ax = plt.subplots(2, 1, figsize=(8,8))
        ax[0].plot(self.old_spectrum[0]/u.GHz, self.old_spectrum[1], alpha=0.8)
        ax[1].plot(self.new_spectrum[0]/u.GHz, self.new_spectrum[1], alpha=0.8)
        ax[0].set(xlabel='Frequency [GHz]', ylabel='Power [dBm]', title='Before optimization')
        ax[1].set(xlabel='Frequency [GHz]', ylabel='Power [dBm]', title='After optimization')
        self.fig = fig

