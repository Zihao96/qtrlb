import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from qtrlb.calibration.calibration import Scan
from qtrlb.utils.waveforms import get_waveform
from qtrlb.processing.processing import gmm_fit, gmm_predict, normalize_population, \
                                        get_readout_fidelity
from qtrlb.processing.fitting import SinModel, ExpSinModel, ExpModel


class DriveAmplitudeScan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 amp_start: float, 
                 amp_stop: float, 
                 amp_points: int, 
                 subspace: str | list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
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
                         prepulse=prepulse,
                         postpulse=postpulse,
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
        
        for i, qubit in enumerate(self.drive_qubits):             
            start = round(self.x_start * 32768)
            start_DRAG = round(start * self.cfg[f'variables.{qubit}/{self.subspace[i]}/DRAG_weight'])
            xinit = f"""
                    move             {start},R4     
                    move             {start_DRAG},R11
            """
            self.sequences[qubit]['program'] += xinit
            
            
    def add_mainpulse(self):
        length = round(self.cfg.variables['common/qubit_pulse_length'] * 1e9) 
        
        for i, qubit in enumerate(self.drive_qubits):
            subspace_dict = self.cfg[f'variables.{qubit}/{self.subspace[i]}']
            
            step = round(self.x_step * 32768)
            step_DRAG = round(step * subspace_dict['DRAG_weight'])
            freq = round((subspace_dict['mod_freq'] + subspace_dict['pulse_detuning']) * 4)
                    
            main = f"""
                 #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     R4,R11
            """  

            main += f"""
                    play             0,1,{length}""" * self.error_amplification_factor
                
            main += f""" 
                    add              R4,{step},R4
                    add              R11,{step_DRAG},R11
            """
            self.sequences[qubit]['program'] += main

        for resonator in self.readout_resonators:
            main = f"""
                 #-----------Main-----------
                    wait             {length*self.error_amplification_factor}
            """            
            self.sequences[resonator]['program'] += main


class RabiScan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float = 0, 
                 length_stop: float = 320e-9, 
                 length_points: int = 81, 
                 subspace: str | list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpSinModel,
                 init_waveform_idx: int = 11):
        
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
                         prepulse=prepulse,
                         postpulse=postpulse,
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
        super().set_waveforms_acquisitions()
        
        for q in self.drive_qubits:
            waveforms = {}
            for i, pulse_length in enumerate(self.x_values):
                pulse_length_ns = round(pulse_length * 1e9)
                if pulse_length_ns == 0: continue
            
                index = i + self.init_waveform_index
                waveforms[f'{index}'] = {"data" : get_waveform(length=pulse_length_ns, 
                                                               shape=self.cfg.variables[f'{q}/pulse_shape']),
                                         "index": index}
            self.sequences[q]['waveforms'].update(waveforms)
        
        
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += xinit
        
        
    def add_mainpulse(self, freq: str = None, gain: str = None):
        """
        Qblox doesn't accept zero length waveform, so we use Label 'end_main' here.
        There is 4ns delay after each Rabi pulse before postpulse/readout.
        It's because the third index of 'play' instruction cannot be register.
        So we cannot set it as a variable, and wait will be separated.
        The parameters freq and gain are left as connector for multidimensional scan.
        We can pass a specific name string of register to it to replace the default values.
        We won't implement DRAG here since we don't have derivative waveform.
        """
        step_ns = round(self.x_step * 1e9)
        
        for i, qubit in enumerate(self.drive_qubits):
            subspace = self.subspace[i]
            subspace_dict = self.cfg[f'variables.{qubit}/{subspace}']
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
            
            self.sequences[qubit]['program'] += main

        for resonator in self.readout_resonators:
            main = f"""
                 #-----------Main-----------
                    jlt              R4,1,@end_main
                    wait             4    
                    wait             R4
        end_main:   add              R4,{step_ns},R4
                    add              R11,1,R11
            """            
            self.sequences[resonator]['program'] += main
            
            
    @property
    def pi_amp(self):
        """
        This property is not error-protected. It just improves convenience.
        We assume linear relation between Rabi Frequency and drive amplitude.
        It should be called only when all qubits are readout.
        """
        pi_amp = {}
        ideal_rabi_freq = 1 / 2 / self.cfg['variables.common/qubit_pulse_length']
        
        for i, q in enumerate(self.drive_qubits):
            r = f'R{q[1:]}'
            fit_rabi_freq = self.fit_result[r].params['freq'].value if hasattr(self, 'fit_result') else 1
            pi_pulse_amp = (ideal_rabi_freq / fit_rabi_freq
                            * self.cfg[f'variables.{q}/{self.subspace[i]}/amp_rabi'])
            pi_amp[q] = pi_pulse_amp
            
        return pi_amp
        
    
class T1Scan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float, 
                 length_stop: float, 
                 length_points: int, 
                 subspace: str | list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
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
                         prepulse=prepulse,
                         postpulse=postpulse,
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += xinit
        
        
    def add_mainpulse(self):
        """
        Because one 'wait' instruction can take no longer than 65534ns, we will divide it by divisor.
        
        Here I use R11 as a deepcopy of R4 and wait a few multiples of divisor first.
        After each wait we substract divisor from R11.
        Then if R11 is smaller than divisor, we wait the remainder.
        """
        pi_pulse = {q: [f'X180_{ss}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        self.add_pulse(pi_pulse, 'T1PIpulse')
        
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += main
        

class RamseyScan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float, 
                 length_stop: float, 
                 length_points: int, 
                 subspace: str | list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
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
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        self.divisor_ns = divisor_ns
        self.artificial_detuning = artificial_detuning

        
    def add_xinit(self):
        """
        We will use R4 for wait time, R12 for angle of VZ gate.
        """
        super().add_xinit()
        
        start_ns = round(self.x_start * 1e9)
        start_ADphase = round(self.x_start * self.artificial_detuning * 1e9)  
        xinit = f"""
                    move             {start_ns},R4            
                    move             {start_ADphase},R12
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += xinit
        
        
    def add_mainpulse(self):
        """   
        All register in sequencer is 32bit, which can only store integer [-2e31,2e31).
        If we do Ramsey with more than 2 phase cycle, then it may cause error.
        Thus we substract 1e9 of R12 when it exceed 1e9, which is one phase cycle of Qblox.
        The wait trick is similar to T1Scan.
        """
        half_pi_pulse = {q: [f'X90_{ss}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        self.add_pulse(half_pi_pulse, 'Ramsey1stHalfPIpulse')
        
        step_ns = round(self.x_step * 1e9)
        step_ADphase = round(self.x_step * self.artificial_detuning * 1e9)  
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
                    
        rmd_wait:   wait             R11
                    set_ph_delta     R12
                    
        end_main:   add              R4,{step_ns},R4
                    add              R12,{step_ADphase},R12
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += main
        
        self.add_pulse(half_pi_pulse, 'Ramsey2ndHalfPIpulse')
        
        
class EchoScan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float, 
                 length_stop: float, 
                 length_points: int, 
                 subspace: str | list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpModel,
                 divisor_ns: int = 65528,
                 echo_type: str = 'CP'):
        
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
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        self.divisor_ns = divisor_ns
        self.echo_type = echo_type

        
    def add_xinit(self):
        super().add_xinit()
        
        start_half_ns = round(self.x_start / 2 * 1e9)
        xinit = f"""
                    move             {start_half_ns},R4            
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += xinit
        
        
    def add_mainpulse(self):
        """
        Here R4 only represent half of the total waiting time.
        """
        half_pi_pulse = {q: [f'X90_{ss}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        self.add_pulse(half_pi_pulse, 'Echo1stHalfPIpulse')
        
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += main
        
        if self.echo_type == 'CP':
            pi_pulse = {q: [f'X180_{ss}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        elif self.echo_type == 'CPMG':
            pi_pulse = {q: [f'Y180_{ss}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        self.add_pulse(pi_pulse, 'EchoPIpulse')
        
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
        for qudit in self.qudits: self.sequences[qudit]['program'] += main
        
        self.add_pulse(half_pi_pulse, 'Echo2ndHalfPIpulse')
        
        
class LevelScan(Scan):
    """ This class assume all qubits start from ground state and we have calibrated PI pulse.
        We then excite them to target level based on self.x_values using PI pulse.
        It's convenient since all Readout-type Scan can inherit this class.
        One can use it to check classification result without reclassifying it.
    """
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 scan_name: str,
                 level_start: int, 
                 level_stop: int,  
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name=scan_name,
                         x_plot_label='Level', 
                         x_plot_unit='arb', 
                         x_start=level_start, 
                         x_stop=level_stop, 
                         x_points=level_stop-level_start+1, 
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit)
        
    
    def check_attribute(self):
        super().check_attribute()
        
        self.x_values = self.x_values.astype(int)  # Useful for correct plot label.
        for r in self.readout_resonators: 
            readout_levels = self.cfg[f'variables.{r}/readout_levels'] 
            assert (readout_levels == self.x_values.tolist()), f'Please check readout levels of {r}!'

        
    def add_xinit(self):
        super().add_xinit()
        
        for qudit in self.qudits: self.sequences[qudit]['program'] += f"""
                    move             {self.x_start},R4            
        """


    def add_mainpulse(self):
        """
        Here we add all PI pulse to our sequence program based on level_stop.
        We will use R4 to represent level and jlt instruction to skip later PI pulse.
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += """
                #-----------Main-----------
                    jlt              R4,1,@end_main    
        """         

        for level in range(self.x_stop):
            self.add_pulse(pulse = {q: [f'X180_{level}{level+1}'] for q in self.drive_qubits},
                           name = f'XPI{level}{level+1}')
            
            for qudit in self.qudits: self.sequences[qudit]['program'] += f"""
                    jlt              R4,{level+2},@end_main    
        """
            
        for qudit in self.qudits: self.sequences[qudit]['program'] += """
        end_main:   add              R4,1,R4    
        """
        
        
class CalibrateClassification(LevelScan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 level_start: int, 
                 level_stop: int,  
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 save_cfg: bool = True):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name='CalibrateClassification',
                         level_start=level_start, 
                         level_stop=level_stop, 
                         prepulse=prepulse,
                         postpulse=postpulse,
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
            data_dict['PopulationCorrected_new'] = np.linalg.solve(data_dict['confusionmatrix_new'],
                                                                   data_dict['PopulationNormalized_new'])
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
        
        

class JustPulse(Scan):
    """ Just a fun way to figure out direction of the Z rotation.
    """
    def __init__(self,
                 cfg,
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 just_pulse: list,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = None):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name='JustPulse',
                         x_plot_label='', 
                         x_plot_unit='arb', 
                         x_start=1, 
                         x_stop=1, 
                         x_points=1, 
                         subspace=None,
                         prepulse=None,
                         postpulse=None,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        self.just_pulse = self.make_it_list(just_pulse)
        
        
    def add_mainpulse(self):
        mainpulse = {q: self.just_pulse for q in self.drive_qubits}
        self.add_pulse(mainpulse, 'Main')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        