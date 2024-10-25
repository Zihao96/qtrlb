import os
import time
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from matplotlib.offsetbox import AnchoredText

import qtrlb.utils.units as u
from qtrlb.config.config import MetaManager
from qtrlb.calibration.calibration import Scan
from qtrlb.utils.waveforms import get_waveform
from qtrlb.processing.fitting import SinModel, ExpSinModel, ExpModel, SpectroscopyModel
from qtrlb.processing.processing import rotate_IQ, gmm_fit, gmm_predict, normalize_population, \
    get_readout_fidelity, plot_corr_matrix, correct_population, two_tone_predict, two_tone_normalize, \
    multitone_predict_sequential, multitone_predict_mask, multitone_normalize, sort_points_by_distance, \
    get_QNDness_matrix, plot_QNDness_matrix




class DriveAmplitudeScan(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 amp_start: float, 
                 amp_stop: float, 
                 amp_points: int, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = SinModel,
                 error_amplification_factor: int = 1):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='Drive_Amplitude',
                         x_plot_label='Drive Amplitude', 
                         x_plot_unit='arb', 
                         x_start=amp_start, 
                         x_stop=amp_stop, 
                         x_points=amp_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         error_amplification_factor=error_amplification_factor)
        
        
    def add_xinit(self):
        """
        Since we want to implement DRAG here and DRAG need a different gain value, \
        we will use register R11 to store the gain for DRAG path.
        """
        super().add_xinit()

        for tone in self.main_tones:
            start = self.gain_translator(self.x_start)
            start_drag = self.gain_translator(self.x_start * self.cfg[f'variables.{tone}/DRAG_weight'])
            xinit = f"""
                    move             {start},R4     
                    move             {start_drag},R11
            """
            self.sequences[tone]['program'] += xinit
            
            
    def add_main(self):
        for tone in self.main_tones:
            tone_dict = self.cfg[f'variables.{tone}']
            
            step = self.gain_translator(self.x_step)
            step_drag = self.gain_translator(self.x_step * tone_dict['DRAG_weight'])
            freq = round((tone_dict['mod_freq'] + tone_dict['pulse_detuning']) * 4)
                    
            main = (f"""
                 #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     R4,R11
            """  
                    
                 + f"""
                    play             0,1,{self.qubit_pulse_length_ns}""" * self.error_amplification_factor

                 + f""" 
                    add              R4,{step},R4
                    add              R11,{step_drag},R11
            """)
            self.sequences[tone]['program'] += main

        for tone in self.rest_tones:
            main = f"""
                 #-----------Main-----------
                    wait             {self.qubit_pulse_length_ns * self.error_amplification_factor}
            """            
            self.sequences[tone]['program'] += main


class Spectroscopy(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 detuning_start: float, 
                 detuning_stop: float, 
                 detuning_points: int, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = SpectroscopyModel,
                 error_amplification_factor: int = 1):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='Spectroscopy',
                         x_plot_label='Pulse Detuning', 
                         x_plot_unit='MHz', 
                         x_start=detuning_start, 
                         x_stop=detuning_stop, 
                         x_points=detuning_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         error_amplification_factor=error_amplification_factor)
        
        
    def add_xinit(self):
        """
        Here R4 stores the modulation frequency of the main tones.
        """
        super().add_xinit()

        for tone in self.main_tones:
            ssb_freq_start = self.x_start + self.cfg[f'variables.{tone}/mod_freq']
            ssb_freq_start_4 = self.frequency_translator(ssb_freq_start)
            
            xinit = f"""
                    move             {ssb_freq_start_4},R4
            """
            self.sequences[tone]['program'] += xinit
            
            
    def add_main(self, gain: str = None, gain_drag: str = None):
        for tone in self.main_tones:
            step = self.frequency_translator(self.x_step)
            if gain is None: gain = round(self.cfg[f'variables.{tone}']['amp_180'] * 32768)
            if gain_drag is None: gain_drag = round(gain * self.cfg[f'variables.{tone}']['DRAG_weight'])

                    
            main = (f"""
                 #-----------Main-----------
                    set_freq         R4
                    set_awg_gain     {gain},{gain_drag}
            """  
                    
                 + f"""
                    play             0,1,{self.qubit_pulse_length_ns}""" * self.error_amplification_factor

                 + f""" 
                    add              R4,{step},R4
            """)
            self.sequences[tone]['program'] += main

        for tone in self.rest_tones:
            main = f"""
                 #-----------Main-----------
                    wait             {self.qubit_pulse_length_ns * self.error_amplification_factor}
            """            
            self.sequences[tone]['program'] += main


    def fit_data(self, x=None, **fitting_kwargs):
        return super().fit_data(
            x=x, 
            t=self.qubit_pulse_length_ns * self.error_amplification_factor * u.ns,
            **fitting_kwargs
        )


class RabiScan(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 length_start: float = 0, 
                 length_stop: float = 320e-9, 
                 length_points: int = 81, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = ExpSinModel,
                 init_waveform_idx: int = 101):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='Rabi',
                         x_plot_label='Pulse Length', 
                         x_plot_unit='ns', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=length_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         init_waveform_idx=init_waveform_idx)
        
        
    def set_waveforms_acquisitions(self):
        """
        RabiScan need each element/x_point to have different pulse length \
        thus different waveforms for Qblox. We will generate them here.
        In case of confliction between index of Rabi waveform and common waveform, \
        we set a minimum index of Rabi waveform here.
        We won't implement DRAG here since it will further limit our waveform length.
        """
        self.check_waveform_length()
        super().set_waveforms_acquisitions(add_special_waveforms=False)
        
        for tone in self.main_tones:
            waveforms = {}
            for i, pulse_length in enumerate(self.x_values):
                pulse_length_ns = round(pulse_length * 1e9)
                if pulse_length_ns == 0: continue
            
                index = i + self.init_waveform_idx
                waveforms[f'{index}'] = {"data" : get_waveform(length=pulse_length_ns, 
                                                               shape=self.cfg[f'variables.{tone}/pulse_shape']),
                                         "index" : index}
            self.sequences[tone]['waveforms'].update(waveforms)
        
        
    def check_waveform_length(self):
        """
        Check the total length of all waveforms since Qblox can only store 16384 samples.
        """
        total_length_ns = np.ceil(np.sum(self.x_values) * 1e9)
        assert total_length_ns < 16384, f'The total pulse length {total_length_ns}ns is too long! \n'\
            'Suggestion: np.linspace(0,200,101), np.linspace(0,320,81), np.linspace(0,600,51).'\
            'np.linspace(320,480,41), np.linspace(600,840,21)'
            

    def add_xinit(self):
        """
        Here R4 will be real pulse length, and R11 is the waveform index.
        So qubit play R11 for drive, resonator wait R4 for sync.
        """
        super().add_xinit()
        
        start_ns = round(self.x_start * 1e9)
        xinit = f"""
                    move             {start_ns},R4
                    move             {self.init_waveform_idx},R11
        """
        for tone in self.tones: self.sequences[tone]['program'] += xinit
        
        
    def add_main(self, freq: str = None, gain: str = None):
        """
        Qblox doesn't accept zero length waveform, so we use Label 'end_main' here.
        There is 4ns delay after each Rabi pulse before post_gate/readout.
        It's because the third index of 'play' instruction cannot be register.
        So we cannot set it as a variable, and wait will be separated.
        The parameters freq and gain are left as connector for multidimensional scan.
        We can pass a specific name string of register to it to replace the default values.
        We won't implement DRAG here since we don't have derivative waveform.
        """
        step_ns = round(self.x_step * 1e9)
        
        for tone in self.main_tones:
            tone_dict = self.cfg[f'variables.{tone}']
            if freq is None: freq = round((tone_dict['mod_freq'] + tone_dict['pulse_detuning']) * 4)
            if gain is None: gain = round(tone_dict['amp_rabi'] * 32768)
                    
            main = f"""
                 #-----------Main-----------
                    jlt              R4,1,@end_main
                    set_freq         {freq}
                    set_awg_gain     {gain},0
                    play             R11,R11,4
                    wait             R4
        end_main:   add              R4,{step_ns},R4
                    add              R11,1,R11
            """  
            
            self.sequences[tone]['program'] += main

        for tone in self.rest_tones:
            main = f"""
                 #-----------Main-----------
                    jlt              R4,1,@end_main
                    wait             4    
                    wait             R4
        end_main:   add              R4,{step_ns},R4
                    add              R11,1,R11
            """            
            self.sequences[tone]['program'] += main
            
            
    @property
    def pi_amp(self):
        """
        Convenience method for daily calibration.
        We assume linear relation between Rabi Frequency and drive amplitude.
        """
        pi_amp = {}
        ideal_rabi_freq = 1 / 2 / self.cfg['variables.common/qubit_pulse_length']
        
        for tone in self.main_tones:
            try:
                rr = 'R' + tone.split('/')[0][1:]
                fit_rabi_freq = self.fit_result[rr].params['freq'].value 
            except AttributeError:
                print('RabiScan: Fitting failed. Please check fitting process.')
                return
            except KeyError:
                # In case we don't readout its resonator but still expect Rabi oscillation.
                # We will just use whatever first element in self.fit_result
                fit_rabi_freq = next(iter(self.fit_result.values())).params['freq'].value 

            pi_pulse_amp = (ideal_rabi_freq / fit_rabi_freq * self.cfg[f'variables.{tone}/amp_rabi'])
            pi_amp[f'{tone}_180'] = pi_pulse_amp
            pi_amp[f'{tone}_90'] = pi_pulse_amp / 2
            
        return pi_amp
        

class DebugRabi(RabiScan):
    """ Make the RabiScan plot both I and Q coordinate in single plot for debugging.
        Work and plot coordinate instead of population even with classification.
    """
    def plot_main(self, text_loc: str = 'lower right'):
        self.figures = {}
        
        for i, rr in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']      
            title = f'{self.datetime_stamp}, {self.scan_name}, {rr}'
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            ylabel = 'Coordinate (Rotated) [a.u.]'
            
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            ax[0].plot(self.x_values / self.x_unit_value, self.measurement[rr]['to_fit'][0], 'k.')
            ax[0].set(xlabel=xlabel, ylabel=f'I-{ylabel}', title=title)
            ax[1].plot(self.x_values / self.x_unit_value, self.measurement[rr]['to_fit'][1], 'k.')
            ax[1].set(xlabel=xlabel, ylabel=f'Q-{ylabel}', title=title)
            
            if self.fit_result[rr] is not None: 
                # Raise resolution of fit result for smooth plot.
                x = np.linspace(self.x_start, self.x_stop, self.x_points * 3)  
                y = self.fit_result[rr].eval(x=x)
                ax[level_index].plot(x / self.x_unit_value, y, 'm-')
                
                # AnchoredText stolen from Ray's code.
                fit_text = '\n'.join([f'{v.name} = {v.value:0.5g}' for v in self.fit_result[rr].params.values()])
                anchored_text = AnchoredText(fit_text, loc=text_loc, prop={'color':'m'})
                ax[level_index].add_artist(anchored_text)

            fig.savefig(os.path.join(self.data_path, f'{rr}.png'))
            self.figures[rr] = fig

    
class T1Scan(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 length_start: float, 
                 length_stop: float, 
                 length_points: int, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = ExpModel,
                 divisor_ns: int = 65528):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='T1',
                         x_plot_label='Wait Length', 
                         x_plot_unit='us', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=length_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         divisor_ns=divisor_ns)

        
    def add_xinit(self):
        super().add_xinit()
        
        start_ns = round(self.x_start * 1e9)
        xinit = f"""
                    move             {start_ns},R4            
        """
        for tone in self.tones: self.sequences[tone]['program'] += xinit
        
        
    def add_main(self):
        """
        Because one 'wait' instruction can take no longer than 65534ns, we will divide it by divisor.
        
        Here I use R11 as a deepcopy of R4 and wait a few multiples of divisor first.
        After each wait we substract divisor from R11.
        Then if R11 is smaller than divisor, we wait the remainder.
        """
        pi_gate = {tone.split('/')[0]: [f'X180_{tone.split("/")[1]}'] for tone in self.main_tones}
        self.add_gate(pi_gate, 'T1PIgate')
        
        step_ns = round(self.x_step * 1e9)
        main = f"""
                #-----------Main-----------
                    jlt              R4,1,@end_main
                    move             R4,R11  
                    nop
                    
        mlt_wait:   jlt              R11,{self.divisor_ns},@rmd_wait
                    wait             {self.divisor_ns}
                    sub              R11,{self.divisor_ns},R11
                    jmp              @mlt_wait
                    
        rmd_wait:   wait             R11
                    
        end_main:   add              R4,{step_ns},R4
        """
        for tone in self.tones: self.sequences[tone]['program'] += main
        

class RamseyScan(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 length_start: float, 
                 length_stop: float, 
                 length_points: int, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = ExpSinModel,
                 divisor_ns: int = 65528,
                 artificial_detuning: float = 0.0,
                 AD_sign: int = None):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='Ramsey',
                         x_plot_label='Wait Length', 
                         x_plot_unit='us', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=length_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         divisor_ns=divisor_ns,
                         artificial_detuning=artificial_detuning)
        
        self.AD_sign = np.sign(self.cfg[f'variables.{self.main_tones[0]}/mod_freq']) if AD_sign is None else AD_sign
        self._artificial_detuning = self.AD_sign * self.artificial_detuning
        # AD generated by 'set_ph_delta' depends on the sign of modulation frequency.

        
    def add_xinit(self):
        """
        We will use R4 for wait time, R12 for angle of VZ gate.
        We will keep R12 always positive, deepcopy it to R13 if the phase is negative.
        Then use 1e9(R14) subtract R13 and store it back to R13.
        It is because the 'set_ph_delta' only take [0, 1e9], not negative values.
        """
        super().add_xinit()
        
        start_ns = round(self.x_start * 1e9)
        start_ADphase = abs(round(self.x_start * self._artificial_detuning * 1e9))  

        xinit = f"""
                    move             {start_ns},R4            
                    move             {start_ADphase},R12
                    move             {int(1e9)}, R14
        """
        for tone in self.tones: self.sequences[tone]['program'] += xinit
        
        
    def add_main(self):
        """   
        All register in sequencer is 32bit, which can only store integer [-2e31, 2e31).
        If we do Ramsey with more than 2 phase cycle, then it may cause error.
        Thus we substract 1e9 of R12 when it exceed 1e9, which is one phase cycle of Qblox.
        The wait trick is similar to T1Scan.
        """
        half_pi_gate = {tone.split('/')[0]: [f'X90_{tone.split("/")[1]}'] for tone in self.main_tones}
        self.add_gate(half_pi_gate, 'Ramsey1stHalfPIgate')
        
        step_ns = round(self.x_step * 1e9)
        step_ADphase = abs(round(self.x_step * self._artificial_detuning * 1e9))
        # Qblox cut one phase circle into 1e9 pieces.
        main = f"""
                #-----------Main-----------
                    jlt              R4,1,@end_main
                    move             R4,R11            
                    
                    jlt              R12,1000000000,@mlt_wait
                    sub              R12,1000000000,R12
                    
        mlt_wait:   jlt              R11,{self.divisor_ns},@rmd_wait
                    wait             {self.divisor_ns}
                    sub              R11,{self.divisor_ns},R11
                    jmp              @mlt_wait
        """

        if self._artificial_detuning >= 0: 
            main += f"""
        rmd_wait:   wait             R11
                    set_ph_delta     R12
                    
        end_main:   add              R4,{step_ns},R4
                    add              R12,{step_ADphase},R12
        """

        else:
            main += f"""
        rmd_wait:   move             R12,R13
                    nop
                    sub              R14,R13,R13
                    wait             R11
                    set_ph_delta     R13
                    
        end_main:   add              R4,{step_ns},R4
                    add              R12,{step_ADphase},R12
        """

        for tone in self.tones: self.sequences[tone]['program'] += main
        
        self.add_gate(half_pi_gate, 'Ramsey2ndHalfPIgate')
        
        
class EchoScan(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 length_start: float, 
                 length_stop: float, 
                 length_points: int, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = ExpSinModel,
                 divisor_ns: int = 65528,
                 artificial_detuning: float = 0.0,
                 AD_sign: int = None,
                 echo_type: str = 'CP',
                 reverse_last_gate: bool = True):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='Echo',
                         x_plot_label='Wait Length', 
                         x_plot_unit='us', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=length_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         divisor_ns=divisor_ns,
                         artificial_detuning=artificial_detuning,
                         echo_type=echo_type,
                         reverse_last_gate=reverse_last_gate)
        
        self.AD_sign = np.sign(self.cfg[f'variables.{self.main_tones[0]}/mod_freq']) if AD_sign is None else AD_sign
        self._artificial_detuning = self.AD_sign * self.artificial_detuning
        # AD generated by 'set_ph_delta' depends on the sign of modulation frequency.
        # For more details about AD, please check RamseyScan.

        
    def add_xinit(self):
        super().add_xinit()
        
        start_half_ns = round(self.x_start / 2 * 1e9)
        start_ADphase = abs(round(self.x_start * self._artificial_detuning * 1e9))
        # We won't split the AD to half since it will be filted by PI pulse in that case.
        # For Echo, it should be implemented all before the second half PI pulse.

        xinit = f"""
                    move             {start_half_ns},R4
                    move             {start_ADphase},R12
                    move             {int(1e9)}, R14
        """
        for tone in self.tones: self.sequences[tone]['program'] += xinit
        
        
    def add_main(self):
        """
        Here R4 only represent half of the total waiting time.
        """
        half_pi_gate = {tone.split('/')[0]: [f'X90_{tone.split("/")[1]}'] for tone in self.main_tones}
        self.add_gate(half_pi_gate, 'Echo1stHalfPIgate')
        
        step_half_ns = round(self.x_step / 2 * 1e9)
        step_ADphase = abs(round(self.x_step * self._artificial_detuning * 1e9))
        # Qblox cut one phase circle into 1e9 pieces.
        main1 = f"""
                #-----------Main1-----------
                    jlt              R4,1,@end_main1
                    move             R4,R11  
                    nop
                    
        mlt_wait1:  jlt              R11,{self.divisor_ns},@rmd_wait1
                    wait             {self.divisor_ns}
                    sub              R11,{self.divisor_ns},R11
                    jmp              @mlt_wait1
                    
        rmd_wait1:  wait             R11
                    
        end_main1:  nop
        """
        for tone in self.tones: self.sequences[tone]['program'] += main1
        
        if self.echo_type == 'CP':
            pi_gate = {tone.split('/')[0]: [f'X180_{tone.split("/")[1]}'] for tone in self.main_tones}
        elif self.echo_type == 'CPMG':
            pi_gate = {tone.split('/')[0]: [f'Y180_{tone.split("/")[1]}'] for tone in self.main_tones}
        self.add_gate(pi_gate, 'EchoPIgate')
        
        main2 = f"""
                #-----------Main2-----------
                    jlt              R4,1,@end_main2
                    move             R4,R11 
                    
                    jlt              R12,1000000000,@mlt_wait2
                    sub              R12,1000000000,R12
                    
        mlt_wait2:  jlt              R11,{self.divisor_ns},@rmd_wait2
                    wait             {self.divisor_ns}
                    sub              R11,{self.divisor_ns},R11
                    jmp              @mlt_wait2
        """

        if self._artificial_detuning >= 0:
            main2 += f"""                    
        rmd_wait2:  wait             R11
                    set_ph_delta     R12
                    
        end_main2:  add              R4,{step_half_ns},R4
                    add              R12,{step_ADphase},R12
        """
            
        else:
            main2 += f"""                    
        rmd_wait2:  move             R12,R13
                    nop
                    sub              R14,R13,R13
                    wait             R11
                    set_ph_delta     R13
                    
        end_main2:  add              R4,{step_half_ns},R4
                    add              R12,{step_ADphase},R12
        """

        for tone in self.tones: self.sequences[tone]['program'] += main2
        
        if self.reverse_last_gate: 
            half_pi_gate = {tone.split('/')[0]: [f'X-90_{tone.split("/")[1]}'] for tone in self.main_tones}
        self.add_gate(half_pi_gate, 'Echo2ndHalfPIgate')
        
        
class LevelScan(Scan):
    """ This class assume all qubits start from ground state and we have calibrated PI gate.
        We then excite them to target level based on self.x_values using PI gate.
        It's convenient since all Readout-type Scan can inherit this class.
        One can use it to check classification result without reclassifying it.

        Note from Zihao(06/01/2023):
        I design the LevelScan and later ReadoutTemplateScan so that they don't need to use attributes \
        like subsapce and main_tones, and these attribute will be kept as default like ['01']. 
        However, the self.tones still matters, see the make_tones_list in LevelScan for more details.
    """
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 scan_name: str,
                 level_start: int, 
                 level_stop: int,  
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None,
                 **attr_kwargs):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name=scan_name,
                         x_plot_label='Level', 
                         x_plot_unit='arb', 
                         x_start=level_start, 
                         x_stop=level_stop, 
                         x_points=level_stop-level_start+1,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         **attr_kwargs)
        
        self.x_values = self.x_values.astype(int)  # Useful for correct plot label.


    def make_tones_list(self):
        """
        Overload the parent method to keep main_tones at Q#/01 as default.
        The self.tones will be determined by the level_stop.
        """
        self.tones = []

        # Determine tones list from level_stop.
        for qubit in self.drive_qubits:
            for level in range(self.x_stop):
                self.tones.append(f'{qubit}/{level}{level+1}')

        self.tones += self.readout_tones

        # Make main_tones default here.
        self.main_tones = [f'{q}/01' for q in self.drive_qubits]

        
    def add_xinit(self):
        super().add_xinit()
        
        for tone in self.tones: self.sequences[tone]['program'] += f"""
                    move             {self.x_start},R4            
        """


    def add_main(self):
        """
        Here we add all PI gate to our sequence program based on level_stop.
        We will use R4 to represent level and jlt instruction to skip later PI gate.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """
                #-----------Main-----------
                    jlt              R4,1,@end_main    
        """         

        for level in range(self.x_stop):
            self.add_gate(gate = {q: [f'X180_{level}{level+1}'] for q in self.drive_qubits},
                          name = f'XPI{level}{level+1}')
            
            for tone in self.tones: self.sequences[tone]['program'] += f"""
                    jlt              R4,{level+2},@end_main    
        """
            
        for tone in self.tones: self.sequences[tone]['program'] += """
        end_main:   add              R4,1,R4    
        """
        
        
class CalibrateClassification(LevelScan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 level_start: int, 
                 level_stop: int,  
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 save_cfg: bool = True,
                 verbose: bool = False,
                 refine_mixture_fitting: bool = True):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='CalibrateClassification',
                         level_start=level_start, 
                         level_stop=level_stop, 
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         save_cfg=save_cfg,
                         verbose=verbose,
                         refine_mixture_fitting=refine_mixture_fitting)
        

    def check_attribute(self):
        super().check_attribute()
        assert self.classification_enable, 'Please turn on classification.'
          
    
    def process_data(self):
        """
        Use the old GMM parameters to process data.
        If the struture of GMM parameters is incompatible, we won't do old GMM processing here.
        It's usually caused by change on readout_levels.
        """
        try:
            return super().process_data()
        
        except Exception as expt:
            if self.heralding_enable: raise expt
            self.plot_populations = lambda: None
            shape = (2, self.n_reps, self.x_points)

            print('Cal: Cannot do processing using old GMM parameters.')

            # Loop over each resonator
            for rr, data_dict in self.measurement.items():
                # Loop over its subtones and process IQ.   
                for subtone, subtone_dict in data_dict.items():
                    # Check whether k is name of subtones. Otherwise if k is process name, we skip it.
                    if not (isinstance(subtone_dict, dict) and 'Heterodyned_readout' in subtone_dict): continue

                    angle = self.cfg[f'process.{rr}/{subtone}/IQ_rotation_angle']
                    subtone_dict['Reshaped_readout'] = np.array(subtone_dict['Heterodyned_readout']).reshape(shape)
                    subtone_dict['IQrotated_readout'] = rotate_IQ(subtone_dict['Reshaped_readout'], angle)

        
    def fit_data(self):
        """
        Here we may already have a normally processed data from self.process_data().
        It will be a reference/comparison from previous classification.
        And we intercept it from 'IQrotated_readout' to do new gmm_fit.
        """
        for rr, data_dict in self.measurement.items(): 
            multitone_IQ_readout = np.concatenate(
                [subtone_dict['IQrotated_readout'] for subtone_dict in data_dict.values()
                 if isinstance(subtone_dict, dict) and 'Heterodyned_readout' in subtone_dict], 
                axis=0)
            
            # First fit GMM parameters for each level separately
            means = np.zeros((self.x_points, multitone_IQ_readout.shape[0]))
            covariances = np.zeros((self.x_points, multitone_IQ_readout.shape[0]))

            for i in range(self.x_points):
                mask = None
                data = multitone_IQ_readout[..., i]
                
                if self.heralding_enable:
                    mask = data_dict['Mask_heralding']
                    data = data[:, mask[:,i] == 0]
                    # Here the multitone_IQ_readout has shape (2*n_tones, n_reps, x_points)
                    # mask has shape (n_reps, x_points)
                
                gmm = gmm_fit(data, n_components=1)
                means[i] = gmm.means_[0]
                covariances[i] = gmm.covariances_[0]
                # Because the default form is one more layer nested.

            # Refit with multi-component model.
            # It's better for poor state preparation or decay during readout.
            if self.refine_mixture_fitting:
                data = multitone_IQ_readout

                if self.heralding_enable: 
                    data = data.reshape(multitone_IQ_readout.shape[0], -1)[:, mask.flatten() == 0]

                gmm = gmm_fit(data, n_components=self.x_points, 
                              refine=True, means=means, covariances=covariances)
                means_new, covariances_new = gmm.means_, gmm.covariances_
                indices = sort_points_by_distance(means_new, means)
                means = means_new[indices]
                covariances = covariances_new[indices]

            # Redo processing and save to measurement dictionary.
            data_dict['means_new'] = means
            data_dict['covariances_new'] = covariances
            data_dict['GMMpredicted_new'] = gmm_predict(multitone_IQ_readout, 
                                                        means=means, covariances=covariances,
                                                        lowest_level=self.x_start)
            data_dict['PopulationNormalized_new'] = normalize_population(data_dict['GMMpredicted_new'],
                                                                         levels=self.x_values,
                                                                         mask=mask)
            data_dict['confusionmatrix_new'] = data_dict['PopulationNormalized_new']
            data_dict['PopulationCorrected_new'] = correct_population(data_dict['PopulationNormalized_new'],
                                                                      data_dict['confusionmatrix_new'],
                                                                      self.cfg.process['corr_method'])
            data_dict['ReadoutFidelity'] = get_readout_fidelity(data_dict['confusionmatrix_new'])
            
            self.cfg[f'process.{rr}/IQ_means'] = means
            self.cfg[f'process.{rr}/IQ_covariances'] = covariances
            self.cfg[f'process.{rr}/corr_matrix'] = data_dict['confusionmatrix_new']
        
        if self.save_cfg: self.cfg.save(verbose=self.verbose)
        
        
    def plot_main(self, dpi: int = 150):
        """
        We expect this function to plot new result with correction.
        And plot_population will give you the previous result along with corrected result.
        So there will be two population plots for each resonator.
        """
        self.figures = {}

        for rr in self.readout_resonators:
            fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=dpi)
            for i, level in enumerate(self.x_values):
                ax[0].plot(self.x_values, self.measurement[rr]['PopulationNormalized_new'][i], 
                           c=f'C{level}', ls='-', marker='.', label=fr'$P_{{{level}}}$')
                ax[1].plot(self.x_values, self.measurement[rr]['PopulationCorrected_new'][i], 
                           c=f'C{level}', ls='-', marker='.', label=fr'$P_{{{level}}}$')

            xlabel = f'{self.x_plot_label}[{self.x_plot_unit}]'
            ax[0].set(xlabel=xlabel, ylabel='Uncorrected populations', ylim=(-0.05, 1.05))
            ax[1].set(xlabel=xlabel, ylabel='Corrected populations', ylim=(-0.05, 1.05))
            ax[0].legend()
            ax[1].legend()
            ax[0].set_title(f'{self.datetime_stamp}, {self.scan_name}, {rr}')
            fig.savefig(os.path.join(self.data_path, f'{rr}_Population_new.png'))
            fig.clear()
            plt.close(fig)

            fig = plot_corr_matrix(self.measurement[rr]['confusionmatrix_new'])
            fig.savefig(os.path.join(self.data_path, f'{rr}_corr_matrix_new.png'))
            self.figures[rr] = fig
        
        
    def plot_IQ(self, dpi: int = 75):
        super().plot_IQ(c_key='GMMpredicted_new', dpi=dpi)
               

class JustGate(Scan):
    """ Simply run the gate sequence user send to it.
        It can be a fun way to figure out direction of the Z rotation.
    """
    def __init__(self,
                 cfg: MetaManager,
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 just_gate: dict[str: list[str]],
                 lengths: list[int],
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 n_seqloops: int = 1,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None,
                 keep_raw: bool = False):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='JustGate',
                         x_plot_label='', 
                         x_plot_unit='arb', 
                         x_start=1, 
                         x_stop=1, 
                         x_points=1, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=None,
                         post_gate=None,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         just_gate=just_gate,
                         lengths=lengths,
                         keep_raw=keep_raw)


    def add_main(self):
        self.add_gate(self.just_gate, 'JustGate', self.lengths)


    def acquire_data(self):
        """
        Keep n_seqloops as 1 and use n_pyloop, we will get all the raw trace.
        The self.measurement[r]['raw_readout'] will have shape (2, n_pyloop, 16384).
        See DACManager.start_sequencer() for more details.
        """
        super().acquire_data(keep_raw=self.keep_raw)


class CalibrateTOF(JustGate):
    def __init__(self, 
                 cfg: MetaManager, 
                 drive_qubits: str, 
                 readout_tones: str):
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits, 
                         readout_tones=readout_tones, 
                         just_gate={drive_qubits: ['I']}, 
                         lengths=None, 
                         n_seqloops=1,
                         keep_raw=True)
        
        self.scan_name = 'CalibrateTOF'


    def plot_main(self, start: int = 0, stop: int = 16384, savefig: bool = True):
        """
        Start and stop set the limit of x axis.
        """
        rr, subtone = self.readout_tones[0].split('/')  # We should use only one readout_tone.
        self.raw_data = np.array(self.measurement[rr][subtone]['raw_readout'])

        t = np.arange(16384)
        I_trace = np.mean(self.raw_data, axis=1)[0]
        Q_trace = np.mean(self.raw_data, axis=1)[1]

        fig, ax = plt.subplots(2, 1, dpi=150)
        ax[0].plot(t[start:stop], I_trace[start:stop])
        ax[1].plot(t[start:stop], Q_trace[start:stop])
        ax[0].set(ylabel='I', title = f'{self.datetime_stamp}, {self.scan_name}, {rr}')
        ax[1].set(ylabel='Q', label='Time[ns]')

        if savefig: fig.savefig(os.path.join(self.data_path, f'{rr}.png'))
        self.figures = {rr: fig}


class CheckBlobShift(CalibrateClassification):
    """ A quick check. Work, but don't take it serious on this scan.
        It just keep doing CalibrateClassification and gmm_fit them.
        It only takes one qubit and one resonator.
    """
    def __init__(self, 
                 cfg: MetaManager, 
                 drive_qubits: str, 
                 readout_tones: str, 
                 n_seqloops: int = 1000):
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits, 
                         readout_tones=readout_tones, 
                         level_start=0,
                         level_stop=1,
                         n_seqloops=n_seqloops, 
                         save_cfg=False)

        self.scan_name = 'CheckBlobShift'
        

    def run(self, n_rounds: int, sleep_seconds: int, experiment_suffix: str = '', n_pyloops: int = 1):
        self.set_running_attributes(experiment_suffix, n_pyloops)
        self.make_sequence() 
        self.save_sequence()
        self.cfg.DAC.implement_parameters(self.tones, self.jsons_path) 

        for i in range(n_rounds):
            self.acquire_data()  # This is really run the thing and return to the IQ data in self.measurement.
            self.process_data()
            self.fit_data()
            self.n_runs += 1
            self.measurements.append(self.measurement)
            print(f'Round {i} finished')
            if not self.n_runs == n_rounds: time.sleep(sleep_seconds)

        self.plot()

    
    def plot(self):
        r = self.readout_resonators[0]  # Assume we check only one readout resonator.

        # The shape of the two arrays should be:
        # means_all.shape = (n_runs, x_points=2, IQ=2)
        # covar_all.shape = (n_runs, x_points=2)
        means_all = np.array([measurement[r]['means_new'] for measurement in self.measurements])
        covar_all = np.array([measurement[r]['covariances_new'] for measurement in self.measurements])

        t = np.arange(self.n_runs)
        fig, ax = plt.subplots(2, 1, dpi=150)
        ax[0].errorbar(t, means_all[:, 0, 0], yerr=covar_all[:, 0], label='|0>, I')
        ax[1].errorbar(t, means_all[:, 0, 1], yerr=covar_all[:, 0], label='|0>, Q')
        ax[0].errorbar(t, means_all[:, 1, 0], yerr=covar_all[:, 1], label='|1>, I')
        ax[1].errorbar(t, means_all[:, 1, 1], yerr=covar_all[:, 1], label='|1>, Q')
        ax[0].set(title='Trace of IQ means with covariances as error bar')
        ax[1].set(xlabel='Number of round')
        ax[0].legend(loc='right', framealpha=0.3)
        ax[1].legend(loc='right', framealpha=0.3)

        self.figures = {r: fig}


class QNDnessCheck(LevelScan):
    """
    An experiment to check QNDness of our readout. Require heralding and pre-GMM-calibration.
    The sequence is following:
    State preparation (Pi-pulses) --> RO --> Ringdown(wait) --> RO

    Note from Zihao(2023/12/26):
    Because of the bad structure of DAC.start_sequencer(), I use a hacky solution here as proof of concept.
    I enforce heralding but do not use it for better state preparation. Instead, it's' one of the two RO above.
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 level_start: int,
                 level_stop: int,
                 ringdown_time: float,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='QNDnessCheck',
                         level_start=level_start,
                         level_stop=level_stop,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         ringdown_time=ringdown_time,
                         ringdown_time_ns=round(ringdown_time / u.ns))


    def check_attribute(self):
        super().check_attribute()
        assert self.heralding_enable, 'QND: Please enable heralding.'


    def add_heralding(self):
        """
        In traditional make_sequence, it comes before subspace_gate, we skip it here.
        """
        return


    def add_readout(self):
        super().add_readout(name='Readout_0', acq_index=0)  # Will become readout in self.measurement
        self.add_wait('RingDown', self.ringdown_time_ns-round(self.cfg.variables['common/tof'] * 1e9))
        super().add_readout(name='Readout_1', acq_index=1)  # will become heralding in self.measurement


    def fit_data(self):
        """
        Calculate QNDness matrix based on processed GMM predicted population.
        We do it here to save QNDness_matrix to measurement.hdf5. 
        """
        for rr in self.readout_resonators:
            self.measurement[rr]['QNDness_matrix'] = get_QNDness_matrix(
                self.measurement[rr]['GMMpredicted_readout'],
                self.measurement[rr]['GMMpredicted_heralding'],
                self.x_values
            )


    def plot_main(self):
        """
        Plot and save the QNDness matrix.
        """
        self.figures = {}
        for rr in self.readout_resonators:
            fig = plot_QNDness_matrix(self.measurement[rr]['QNDness_matrix'])
            fig.savefig(os.path.join(self.data_path, f'{rr}_QNDness_matrix.png'))
            self.figures[rr] = fig


    def plot_IQ(self, IQ_key: str = 'IQrotated_readout', c_key: str = 'GMMpredicted_readout', dpi: int = 75):
        """
        We do a hack here by set heralding enable to false and plot_IQ before change it back.
        Unfortunately, we cannot plot and save two readout IQ at same time since their filename will overlap.
        """
        self.heralding_enable = False
        super().plot_IQ(IQ_key, c_key, dpi=dpi)
        self.heralding_enable = True


class TwoToneROCalibration(LevelScan):
    """ This class is designed for calibrating classification of twotone readout.
        We will generate new GMM parameters with single tone corr_matrix for each tone.
        We will also use them to generate self.twotone_corr_matrix whose shape depends on data process method.
        The readout levels should has same length as readout resonators.
        Each sublist in this list is the readout levels corresponds to that resonators.

        Note from Zihao(09/12/2023):
        The larger matrix will be returned but not saved to yaml file.
        This is because I want to do it fast and I haven't get a nice structure for process yaml.
    """
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: list[str],
                 level_start: int, 
                 level_stop: int,
                 readout_levels_dict: dict[str: list[int]],  
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 save_cfg: bool = True):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='TwoToneROCalibration',
                         level_start=level_start, 
                         level_stop=level_stop, 
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         readout_levels_dict=readout_levels_dict,
                         save_cfg=save_cfg)


    def check_attribute(self):
        super().check_attribute()
        assert self.classification_enable, 'Please turn on classification.'
        assert not self.heralding_enable, 'This Scan do not support heralding yet.'
        assert self.customized_data_process is not None, 'Please specify customized data process.'
        assert len(self.readout_levels_dict) == len(self.readout_tones) == 2, \
                'Please specify two resonators and their own readout levels.'


    def process_data(self):
        """
        Overload parent method since we cannot use those customized process here.
        It can become a possible position to implement heralding for this scan.
        """
        return
    

    def fit_data(self):
        """
        Fit GMM parameters for each tone and calculate corr_matrix.
        Also generate the self.twotone_corr_matrix.
        Please see CalibrateClassification.fit_data() for reference.
        Or maybe try ProcessManager first if you haven't read it.
        """
        shape = (2, self.n_reps, self.x_points)

        for r, readout_levels in self.readout_levels_dict.items():
            # Initial process
            data_dict = self.measurement[r]
            data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
            data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Reshaped_readout'], 
                                                       angle=self.cfg.process[f'{r}/IQ_rotation_angle'])
            
            # GMM fitting
            means = np.zeros((len(readout_levels), 2))
            covariances = np.zeros(len(readout_levels))

            for i, l in enumerate(readout_levels):
                data = data_dict['IQrotated_readout'][..., l]
                mean, covariance = gmm_fit(data, n_components=1)
                means[i] = mean[0]
                covariances[i] = covariance[0]
                # Because the default form is one more layer nested.

            # Using fitting result to re-predict state.
            data_dict['means_new'] = means
            data_dict['covariances_new'] = covariances
            data_dict['GMMpredicted_new'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                        means=means, 
                                                        covariances=covariances,
                                                        lowest_level=readout_levels[0])

            # Find single tone corr_matrix. Not all data should be used here!
            corr_matrix = normalize_population(data_dict['GMMpredicted_new'][:, readout_levels],
                                               levels=readout_levels)

            # Set it to self.measurement and cfg.
            data_dict['confusionmatrix_new'] = corr_matrix
            data_dict['ReadoutFidelity'] = get_readout_fidelity(corr_matrix)
            self.cfg[f'process.{r}/IQ_means'] = means
            self.cfg[f'process.{r}/IQ_covariances'] = covariances
            self.cfg[f'process.{r}/corr_matrix'] = corr_matrix

        if self.save_cfg: self.cfg.save()
        tone_0, tone_1 = self.readout_tones

        if self.customized_data_process == 'two_tone_readout_mask':
            # Predict result and generate mask for confliction.
            twotonepredicted_readout, mask_twotone = two_tone_predict(
                self.measurement[tone_0]['GMMpredicted_new'],
                self.measurement[tone_1]['GMMpredicted_new'],
                self.readout_levels_dict[tone_0],
                self.readout_levels_dict[tone_1]
            )
            self.twotone_corr_matrix = normalize_population(
                twotonepredicted_readout,
                levels=np.union1d(self.readout_levels_dict[tone_0], self.readout_levels_dict[tone_1]),
                mask=mask_twotone
            )
            twotone_fidelity = get_readout_fidelity(self.twotone_corr_matrix)
            print(f'TTROCal: TwoTone Readout Fidelity: {twotone_fidelity}')
            
            # Save it to self.measurement
            for r, data_dict in self.measurement.items():
                data_dict['TwoTonePredicted_readout'] = twotonepredicted_readout
                data_dict['Mask_twotone'] = mask_twotone
                data_dict['TwoTone_corr_matrix'] = self.twotone_corr_matrix
                data_dict['TwoTone_ReadoutFidelity'] = twotone_fidelity

        elif self.customized_data_process == 'two_tone_readout_corr':
            self.twotone_corr_matrix = two_tone_normalize(
                self.measurement[tone_0]['GMMpredicted_new'],
                self.measurement[tone_1]['GMMpredicted_new'],
                self.readout_levels_dict[tone_0],
                self.readout_levels_dict[tone_1]
            )
            # Save it to self.measurement
            for r, data_dict in self.measurement.items():
                data_dict['TwoTone_corr_matrix'] = self.twotone_corr_matrix

        else:
            raise NotImplementedError(f'TTROCal: Cannot fit data for {self.customized_data_process} process.')
        

    def plot(self):
        """
        Here we just plot the self.twoton_corr_matrix and IQ.
        """
        self.figure = plot_corr_matrix(self.twotone_corr_matrix)
        self.figure.savefig(os.path.join(self.data_path, f'TwoTone_corr_matrix.png'))
        super().plot_IQ(c_key='GMMpredicted_new')


class MultitoneROCalibration(LevelScan):
    """ This class is designed for calibrating classification of multitone readout.
        We will generate new GMM parameters with single tone and corr_matrix for each tone.
        We will also use them to generate self.multitone_corr_matrix whose shape depends on data process method.
        When call self.run(), process_kwargs is necessary here. 
        User need to check readout levels in variables.yaml before run this scan.

        Example of readout_levels_dict:
        {
        'R4a': [0,1,2,3], 'R4b': [3,4,5,6], 'R4c': [6,7,8]
        }

        Note from Zihao(11/07/2023):
        If we don't have previous multitone_corr_matrix, we should pass in identity or zeros if you like.
        It's for making the process_data work.
        If we have previous multitone_corr_matrix, then process_data will give old result and heralding mask.

        The multitone_corr_matrix matrix will be returned but not saved to yaml file.
        This is because I want to do it fast and I haven't get a nice structure for process yaml.
        Besides, it should allowed to readout more resonators than those we want to fit.
        But I haven't test it.
    """
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_tones: list[str],
                 level_start: int, 
                 level_stop: int,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 save_cfg: bool = True,
                 refine_mixture_fitting: bool = False):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='MultitoneROCalibration',
                         level_start=level_start, 
                         level_stop=level_stop, 
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         save_cfg=save_cfg,
                         refine_mixture_fitting=refine_mixture_fitting)
        

    def check_attribute(self):
        super().check_attribute()
        assert self.classification_enable, 'Please turn on classification.'
        assert self.customized_data_process is not None, 'Please specify customized data process.'
    

    def fit_data(self):
        """
        Fit GMM parameters for each tone and calculate corr_matrix.
        Also generate the self.multitone_corr_matrix.
        Please see CalibrateClassification.fit_data() for reference.
        Or maybe try ProcessManager first if you haven't read it.
        """
        # Fit new GMM parameter and get corr_matrix for each single tone.
        for r, data_dict in self.measurement.items():
            # Fit GMM parameters for each level separately
            readout_levels = self.cfg.variables[f'{r}/readout_levels']
            means = np.zeros((len(readout_levels), 2))
            covariances = np.zeros(len(readout_levels))

            for i, l in enumerate(readout_levels):
                mask_heralding = None
                data = data_dict['IQrotated_readout'][..., l]
                
                if self.heralding_enable:
                    mask_heralding = data_dict['Mask_heralding']

                    # In mask strategy, data need to have no contradiction and pass heralding test.
                    if self.customized_data_process == 'multitone_readout_mask':
                        mask_heralding = mask_heralding | data_dict['Mask_multitone_heralding']
                        
                    data = data[:, mask_heralding[:, l] == 0]
                    # Here the data_dict['IQrotated_readout'] has shape (2, n_reps, x_points)
                    # All mask have shape (n_reps, x_points)

                mean, covariance = gmm_fit(data, n_components=1)
                means[i] = mean[0]
                covariances[i] = covariance[0]
                # Because the default form is one more layer nested.


            # Refit with multi-component model.
            # It's better for poor state preparation or decay during readout.
            if self.refine_mixture_fitting:
                data = data_dict['IQrotated_readout'][..., readout_levels]
                if self.heralding_enable: data = data.reshape(2, -1)[:, mask_heralding[:,readout_levels].flatten() == 0]

                means_new, covariances_new = gmm_fit(data, n_components=len(readout_levels))
                indices = sort_points_by_distance(means_new, means)
                means = means_new[indices]
                covariances = covariances_new[indices]


            # Using fitting result to re-predict state.
            data_dict['means_new'] = means
            data_dict['covariances_new'] = covariances
            data_dict['GMMpredicted_new'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                        means=means, 
                                                        covariances=covariances,
                                                        lowest_level=readout_levels[0])

            # Find single tone corr_matrix. Not all data should be used here!
            corr_matrix = normalize_population(data_dict['GMMpredicted_new'][:, readout_levels],
                                               levels=readout_levels)

            # Set it to self.measurement and cfg.
            data_dict['confusionmatrix_new'] = corr_matrix
            single_tone_fidelity = get_readout_fidelity(corr_matrix)
            data_dict['ReadoutFidelity'] = single_tone_fidelity
            print(f'MtROCal: {r} Readout Fidelity: {single_tone_fidelity}')

            self.cfg[f'process.{r}/IQ_means'] = means
            self.cfg[f'process.{r}/IQ_covariances'] = covariances
            self.cfg[f'process.{r}/corr_matrix'] = corr_matrix

        if self.save_cfg: self.cfg.save()

        data_levels_tuple = (
            (data_dict['GMMpredicted_new'], self.cfg.process[f'{r}/readout_levels']) 
            for r, data_dict in self.measurement.items()
        )

        # Use new GMM predicted result to calculate new multitone_corr_matrix
        if self.customized_data_process == 'multitone_readout_sequential':
            multitonepredicted_readout = multitone_predict_sequential(*data_levels_tuple)
            self.multitone_corr_matrix = normalize_population(
                multitonepredicted_readout,
                levels=self.x_values,
                mask=mask_heralding
            )
            multitone_fidelity = get_readout_fidelity(self.multitone_corr_matrix)
            print(f'MtROCal: Multitone Readout Fidelity: {multitone_fidelity}')

            # Save it to self.measurement
            for r, data_dict in self.measurement.items():
                data_dict['MultitonePredicted_readout'] = multitonepredicted_readout
                data_dict['Multitone_corr_matrix'] = self.multitone_corr_matrix
                data_dict['Multitone_ReadoutFidelity'] = multitone_fidelity

        elif self.customized_data_process == 'multitone_readout_mask':
            multitonepredicted_readout, mask_multitone_readout = multitone_predict_mask(*data_levels_tuple)
            if mask_heralding is None: mask_heralding = 0
            mask_union = mask_heralding | mask_multitone_readout
            self.multitone_corr_matrix = normalize_population(
                multitonepredicted_readout,
                levels=self.x_values,
                mask=mask_union
            )
            multitone_fidelity = get_readout_fidelity(self.multitone_corr_matrix)
            print(f'MtROCal: Multitone Readout Fidelity: {multitone_fidelity}')

            # Save it to self.measurement
            for r, data_dict in self.measurement.items():
                data_dict['MultitonePredicted_readout'] = multitonepredicted_readout
                data_dict['Mask_multitone_readout'] = mask_multitone_readout
                data_dict['Mask_union'] = mask_union
                data_dict['Multitone_corr_matrix'] = self.multitone_corr_matrix
                data_dict['Multitone_ReadoutFidelity'] = multitone_fidelity

        elif self.customized_data_process == 'multitone_readout_corr':
            self.multitone_corr_matrix = multitone_normalize(*data_levels_tuple, mask=mask_heralding)

            # Save it to self.measurement
            for r, data_dict in self.measurement.items():
                data_dict['Multitone_corr_matrix'] = self.multitone_corr_matrix

        else:
            raise NotImplementedError(f'MtROCal: Cannot fit data for {self.customized_data_process} process.')


    def plot(self):
        """
        Here we just plot the self.twoton_corr_matrix and IQ.
        """
        self.figure = plot_corr_matrix(self.multitone_corr_matrix)
        self.figure.savefig(os.path.join(self.data_path, f'Multitone_corr_matrix.png'))
        super().plot_IQ(c_key='GMMpredicted_new')

