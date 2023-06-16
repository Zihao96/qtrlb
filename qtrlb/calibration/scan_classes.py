import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from lmfit import Model
from qtrlb.config.config import MetaManager
from qtrlb.calibration.calibration import Scan
from qtrlb.utils.waveforms import get_waveform
from qtrlb.processing.processing import gmm_fit, gmm_predict, normalize_population, \
                                        get_readout_fidelity, correct_population
from qtrlb.processing.fitting import SinModel, ExpSinModel, ExpModel


class DriveAmplitudeScan(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_resonators: str | list[str],
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
                         readout_resonators=readout_resonators,
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
                         fitmodel=fitmodel)
        
        self.error_amplification_factor = error_amplification_factor
        
        
    def add_xinit(self):
        """
        Since we want to implement DRAG here and DRAG need a different gain value, \
        we will use register R11 to store the gain for DRAG path.
        """
        super().add_xinit()

        for tone in self.main_tones:
            start = self.gain_translator(self.x_start)
            start_DRAG = self.gain_translator(self.x_start * self.cfg[f'variables.{tone}/DRAG_weight'])
            # start = round(self.x_start * 32768)
            # start_DRAG = round(start * self.cfg[f'variables.{tone}/DRAG_weight'])
            xinit = f"""
                    move             {start},R4     
                    move             {start_DRAG},R11
            """
            self.sequences[tone]['program'] += xinit
            
            
    def add_main(self):
        for tone in self.main_tones:
            subspace_dict = self.cfg[f'variables.{tone}']
            
            step = self.gain_translator(self.x_step)
            step_DRAG = self.gain_translator(self.x_step * subspace_dict['DRAG_weight'])
            freq = round((subspace_dict['mod_freq'] + subspace_dict['pulse_detuning']) * 4)
                    
            main = (f"""
                 #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     R4,R11
            """  
                    
                 + f"""
                    play             0,1,{self.qubit_pulse_length_ns}""" * self.error_amplification_factor

                 + f""" 
                    add              R4,{step},R4
                    add              R11,{step_DRAG},R11
            """)
            self.sequences[tone]['program'] += main

        for tone in self.rest_tones:
            main = f"""
                 #-----------Main-----------
                    wait             {self.qubit_pulse_length_ns * self.error_amplification_factor}
            """            
            self.sequences[tone]['program'] += main


class RabiScan(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_resonators: str | list[str],
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
                         readout_resonators=readout_resonators,
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
                         fitmodel=fitmodel)
        
        self.init_waveform_index = init_waveform_idx
        
        
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
            
                index = i + self.init_waveform_index
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
                    move             {self.init_waveform_index},R11
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
            subspace_dict = self.cfg[f'variables.{tone}']
            if freq is None: freq = round((subspace_dict['mod_freq'] + subspace_dict['pulse_detuning']) * 4)
            if gain is None: gain = round(subspace_dict['amp_rabi'] * 32768)
                    
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
        This property is not error-protected. It just improves convenience.
        We assume linear relation between Rabi Frequency and drive amplitude.
        It should be called only when all qubits are readout.
        """
        pi_amp = {}
        ideal_rabi_freq = 1 / 2 / self.cfg['variables.common/qubit_pulse_length']
        
        for tone in self.main_tones:
            r = 'R' + tone.split('/')[0][1:]
            fit_rabi_freq = self.fit_result[r].params['freq'].value if hasattr(self, 'fit_result') else 1
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
        
        for i, r in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{r}/lowest_readout_levels']      
            title = f'{self.datetime_stamp}, {self.scan_name}, {r}'
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            ylabel = 'Coordinate (Rotated) [a.u.]'
            
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            ax[0].plot(self.x_values / self.x_unit_value, self.measurement[r]['to_fit'][0], 'k.')
            ax[0].set(xlabel=xlabel, ylabel=f'I-{ylabel}', title=title)
            ax[1].plot(self.x_values / self.x_unit_value, self.measurement[r]['to_fit'][1], 'k.')
            ax[1].set(xlabel=xlabel, ylabel=f'Q-{ylabel}', title=title)
            
            if self.fit_result[r] is not None: 
                # Raise resolution of fit result for smooth plot.
                x = np.linspace(self.x_start, self.x_stop, self.x_points * 3)  
                y = self.fit_result[r].eval(x=x)
                ax[level_index].plot(x / self.x_unit_value, y, 'm-')
                
                # AnchoredText stolen from Ray's code.
                fit_text = '\n'.join([f'{v.name} = {v.value:0.5g}' for v in self.fit_result[r].params.values()])
                anchored_text = AnchoredText(fit_text, loc=text_loc, prop={'color':'m'})
                ax[level_index].add_artist(anchored_text)

            fig.savefig(os.path.join(self.data_path, f'{r}.png'))
            self.figures[r] = fig

    
class T1Scan(Scan):
    def __init__(self, 
                 cfg: MetaManager,  
                 drive_qubits: str | list[str],
                 readout_resonators: str | list[str],
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
                         readout_resonators=readout_resonators,
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
                         fitmodel=fitmodel)
        
        self.divisor_ns = divisor_ns

        
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
                 readout_resonators: str | list[str],
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
                 artificial_detuning: float = 0.0):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
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
                         fitmodel=fitmodel)
        
        self.divisor_ns = divisor_ns
        self.artificial_detuning = artificial_detuning
        self._artificial_detuning = -1 * self.artificial_detuning
        # It's because the AD is opposite of the natural direction of 'set_ph_delta'.

        
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
                 readout_resonators: str | list[str],
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
                 divisor_ns: int = 65528,
                 echo_type: str = 'CP',
                 reverse_last_gate: bool = True):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
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
                         fitmodel=fitmodel)
        
        self.divisor_ns = divisor_ns
        self.echo_type = echo_type
        self.reverse_last_gate = reverse_last_gate

        
    def add_xinit(self):
        super().add_xinit()
        
        start_half_ns = round(self.x_start / 2 * 1e9)
        xinit = f"""
                    move             {start_half_ns},R4            
        """
        for tone in self.tones: self.sequences[tone]['program'] += xinit
        
        
    def add_main(self):
        """
        Here R4 only represent half of the total waiting time.
        """
        half_pi_gate = {tone.split('/')[0]: [f'X90_{tone.split("/")[1]}'] for tone in self.main_tones}
        self.add_gate(half_pi_gate, 'Echo1stHalfPIgate')
        
        step_half_ns = round(self.x_step / 2 * 1e9)
        main = f"""
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
        for tone in self.tones: self.sequences[tone]['program'] += main
        
        if self.echo_type == 'CP':
            pi_gate = {tone.split('/')[0]: [f'X180_{tone.split("/")[1]}'] for tone in self.main_tones}
        elif self.echo_type == 'CPMG':
            pi_gate = {tone.split('/')[0]: [f'Y180_{tone.split("/")[1]}'] for tone in self.main_tones}
        self.add_gate(pi_gate, 'EchoPIgate')
        
        main = f"""
                #-----------Main2-----------
                    jlt              R4,1,@end_main2
                    move             R4,R11 
                    nop
                    
        mlt_wait2:  jlt              R11,{self.divisor_ns},@rmd_wait2
                    wait             {self.divisor_ns}
                    sub              R11,{self.divisor_ns},R11
                    jmp              @mlt_wait2
                    
        rmd_wait2:  wait             R11
                    
        end_main2:  add              R4,{step_half_ns},R4
        """
        for tone in self.tones: self.sequences[tone]['program'] += main
        
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
                 readout_resonators: str | list[str],
                 scan_name: str,
                 level_start: int, 
                 level_stop: int,  
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name=scan_name,
                         x_plot_label='Level', 
                         x_plot_unit='arb', 
                         x_start=level_start, 
                         x_stop=level_stop, 
                         x_points=level_stop-level_start+1,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit)
        
    
    def check_attribute(self):
        super().check_attribute()
        
        self.x_values = self.x_values.astype(int)  # Useful for correct plot label.
        for r in self.readout_resonators: 
            readout_levels = self.cfg[f'variables.{r}/readout_levels'] 
            assert (readout_levels == self.x_values.tolist()), f'Please check readout levels of {r}!'


    def make_tones_list(self):
        """
        Overload the parent method to keep main_tones at Q#/01 as default.
        The self.tones will be determined by the level_stop.
        """
        self.tones = []
        self.tones_ = []  # Replace all slash by underscroll. Just for convenience.

        # Determine tones list from level_stop.
        for qubit in self.drive_qubits:
            for level in range(self.x_stop):
                self.tones.append(f'{qubit}/{level}{level+1}')
                self.tones_.append(f'{qubit}_{level}{level+1}')

        self.tones += self.readout_resonators
        self.tones_ += self.readout_resonators

        # Make main_tones default here.
        self.main_tones = [f'{q}/01' for q in self.drive_qubits]
        self.main_tones_ = [main_tone.replace('/', '_') for main_tone in self.main_tones]
        self.rest_tones = [tone for tone in self.tones if tone not in self.main_tones]

        
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
                 readout_resonators: str | list[str],
                 level_start: int, 
                 level_stop: int,  
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 save_cfg: bool = True):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name='CalibrateClassification',
                         level_start=level_start, 
                         level_stop=level_stop, 
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops)
        
        self.save_cfg = save_cfg
        assert self.classification_enable, 'Please turn on classification.'
          
        
    def fit_data(self):
        """
        Here we should already have a normally processed data.
        It will be a reference/comparison from previous classification.
        And we intercept it from 'IQrotated_readout' to do new gmm_fit.
        """
        for r, data_dict in self.measurement.items():
            means = np.zeros((self.x_points, 2))
            covariances = np.zeros(self.x_points)
            
            for i in range(self.x_points):
                mask = None
                data = data_dict['IQrotated_readout'][..., i]
                
                if self.heralding_enable:
                    mask = data_dict['Mask_heralding']
                    data = data[:, mask[:,i] == 0]
                    # Here the data_dict['IQrotated_readout'] has shape (2, n_reps, x_points)
                    # mask has shape (n_reps, x_points)
                
                mean, covariance = gmm_fit(data, n_components=1)
                means[i] = mean[0]
                covariances[i] = covariance[0]
                # Because the default form is one more layer nested.
                
            data_dict['means_new'] = means
            data_dict['covariances_new'] = covariances
            data_dict['GMMpredicted_new'] = gmm_predict(data_dict['IQrotated_readout'], 
                                                        means=means, covariances=covariances)
            data_dict['PopulationNormalized_new'] = normalize_population(data_dict['GMMpredicted_new'],
                                                                         n_levels=self.x_points,
                                                                         mask=mask)
            data_dict['confusionmatrix_new'] = data_dict['PopulationNormalized_new']
            data_dict['PopulationCorrected_new'] = correct_population(data_dict['PopulationNormalized_new'],
                                                                      data_dict['confusionmatrix_new'])
            data_dict['ReadoutFidelity'] =  get_readout_fidelity(data_dict['confusionmatrix_new'])
            
            self.cfg[f'process.{r}/IQ_means'] = means
            self.cfg[f'process.{r}/IQ_covariances'] = covariances
            self.cfg[f'process.{r}/corr_matrix'] = data_dict['PopulationNormalized_new']
        
        if self.save_cfg: self.cfg.save()
        
        
    def plot_main(self):
        """
        We expect this function to plot new result with correction.
        And plot_all_population will give you the previous result along with corrected result.
        So there will be four plots for each resonator in total.
        """
        for r in self.readout_resonators:
            title = f'Uncorrected probability, {self.scan_name}, {r}'
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            ylabel = 'Probability'
            
            fig, ax = plt.subplots(1, 1, dpi=150)
            for i, level in enumerate(self.x_values):
                ax.plot(self.x_values, self.measurement[r]['PopulationNormalized_new'][i], 
                        c=f'C{level}', ls='-', marker='.', label=fr'$P_{{{level}}}$')
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title, ylim=(-0.05,1.05))
            plt.legend()
            fig.savefig(os.path.join(self.data_path, f'{r}_PopulationUncorrected_new.png'))
            
            
            title = f'Corrected probability, {self.scan_name}, {r}'
            fig, ax = plt.subplots(1, 1, dpi=150)
            for i, level in enumerate(self.x_values):
                ax.plot(self.x_values, self.measurement[r]['PopulationCorrected_new'][i], 
                        c=f'C{level}', ls='-', marker='.', label=fr'$P_{{{level}}}$')
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title, ylim=(-0.05,1.05))
            plt.legend()
            fig.savefig(os.path.join(self.data_path, f'{r}_PopulationCorrected_new.png'))
            plt.close(fig)
        
        
    def plot_IQ(self):
        super().plot_IQ(c_key='GMMpredicted_new')
               

class JustGate(Scan):
    """ Simply run the gate sequence user send to it.
        It can be a fun way to figure out direction of the Z rotation.
    """
    def __init__(self,
                 cfg: MetaManager,
                 drive_qubits: str | list[str],
                 readout_resonators: str | list[str],
                 just_gate: dict[str: list[str]],
                 lengths: list[int],
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
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
                         fitmodel=fitmodel)
        
        self.just_gate = just_gate
        self.lengths = lengths


    def add_main(self):
        self.add_gate(self.just_gate, 'JustGate', self.lengths)


class CalibrateTOF(JustGate):
    def __init__(self, 
                 cfg: MetaManager, 
                 drive_qubits: str, 
                 readout_resonators: str):
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits, 
                         readout_resonators=readout_resonators, 
                         just_gate={drive_qubits: ['I']}, 
                         lengths=None, 
                         n_seqloops=1)
        
        self.scan_name = 'CalibrateTOF'
        

    def run(self, experiment_suffix: str = '', n_pyloops: int = 1000):
        super().run(experiment_suffix, n_pyloops)


    def acquire_data(self):
        super().acquire_data(keep_raw=True)


    def plot_main(self, start: int = 0, stop: int = 16384, savefig: bool = True):
        """
        Start and stop set the limit of x axis.
        """
        r = self.readout_resonators[0]  # We should only use one resonator.
        self.raw_data = np.array(self.measurement[r]['raw_readout'])

        t = np.arange(16384)
        I_trace = np.mean(self.raw_data, axis=1)[0]
        Q_trace = np.mean(self.raw_data, axis=1)[1]

        fig, ax = plt.subplots(2, 1, dpi=150)
        ax[0].plot(t[start:stop], I_trace[start:stop])
        ax[1].plot(t[start:stop], Q_trace[start:stop])
        ax[0].set(ylabel='I', title = f'{self.datetime_stamp}, {self.scan_name}, {r}')
        ax[1].set(ylabel='Q', label='Time[ns]')

        if savefig: fig.savefig(os.path.join(self.data_path, f'{r}.png'))
        self.figures = {r: fig}


class CheckBlobShift(CalibrateClassification):
    """ A quick check. Work, but don't take it serious on this scan.
        It just keep doing CalibrateClassification and gmm_fit them.
        It only takes one qubit and one resonator.
    """
    def __init__(self, 
                 cfg: MetaManager, 
                 drive_qubits: str, 
                 readout_resonators: str, 
                 n_seqloops: int = 1000):
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits, 
                         readout_resonators=readout_resonators, 
                         level_start=0,
                         level_stop=1,
                         n_seqloops=n_seqloops, 
                         save_cfg=False)

        self.scan_name = 'CheckBlobShift'
        

    def run(self, n_rounds: int, sleep_seconds: int, experiment_suffix: str = '', n_pyloops: int = 1):
        self.experiment_suffix = experiment_suffix
        self.n_pyloops = n_pyloops
        self.n_reps = self.n_seqloops * self.n_pyloops
        self.attrs = {k: v for k, v in self.__dict__.items() if not k.startswith(('cfg', 'measurement'))}
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
        r = self.readout_resonators[0]

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