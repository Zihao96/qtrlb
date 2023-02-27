import numpy as np
from lmfit import Model
from qtrlb.calibration.calibration import Scan
from qtrlb.utils.waveforms import get_waveform
from qtrlb.processing.fitting import SinModel, ExpSinModel, ExpModel


class DriveAmplitudeScan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 amp_start: float, 
                 amp_stop: float, 
                 amp_points: int, 
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = SinModel,
                 error_amplification_factor: int = 1):
        self.error_amplification_factor = error_amplification_factor
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='Drive_Amplitude',
                         x_label_plot='Drive Amplitude', 
                         x_unit_plot='[a.u].', 
                         x_start=amp_start, 
                         x_stop=amp_stop, 
                         x_points=amp_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        
    def add_initvalues(self):
        for i, qubit in enumerate(self.drive_qubits):
            start = round(self.x_start * 32768)
            initparameter = f"""
                    move             {start},R4            
            """
            self.sequences[qubit]['program'] += initparameter
            
            
    def add_mainpulse(self):
        length = round(self.cfg.variables['common/qubit_pulse_length'] * 1e9) 
        
        for i, qubit in enumerate(self.drive_qubits):
            step = round(self.x_step * 32768)
            subspace = self.subspace[i]
            freq = round(self.cfg.variables[f'{qubit}/{subspace}/mod_freq'] * 4)   
                    
            main = f"""
                 #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     R4,R4
            """  

            main += f"""
                    play             0,0,{length}""" * self.error_amplification_factor
                
            main += f""" 
                    add              R4,{step},R4
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
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpSinModel,
                 init_waveform_idx: int = 11):
        self.init_waveform_index = init_waveform_idx
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='Rabi',
                         x_label_plot='Pulse Length', 
                         x_unit_plot='[ns].', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=length_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        
    def set_waveforms_acquisitions(self):
        """
        RabiScan need each element/x_point to have different pulse length 
        thus different waveforms for Qblox. We will generate them here.
        In case of confliction between index of Rabi waveform and common waveform,
        we set a minimum index of Rabi waveform here.
        """
        self.check_waveform_length()
        super().set_waveforms_acquisitions()
        
        for q in self.drive_qubits:
            waveforms = {}
            for i in range(self.x_points):
                pulse_length_ns = round(self.x_values[i] * 1e9)
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
            

    def add_initvalues(self):
        """
        Here R4 will be real pulse length, and R11 is the waveform index.
        So qubit play R11 for drive, resonator wait R4 for sync.
        """
        start_ns = round(self.x_start * 1e9)
        initparameter = f"""
                    move             {start_ns},R4
                    move             {self.init_waveform_index},R11
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += initparameter
        
        
    def add_mainpulse(self):
        """
        Qblox doesn't accept zero length waveform, so we use Label 'end_main' here.
        There is 4ns delay after each Rabi pulse before postpulse/readout.
        It's because the third index of 'play' instruction cannot be register.
        So we cannot set it as a variable, and wait will be separated.
        """
        step_ns = round(self.x_step * 1e9)
        
        for i, qubit in enumerate(self.drive_qubits):
            subspace = self.subspace[i]
            freq = round(self.cfg.variables[f'{qubit}/{subspace}/mod_freq'] * 4) 
            gain = round(self.cfg.variables[f'{qubit}/{subspace}/amp_rabi'] * 32768)
                    
            main = f"""
                 #-----------Main-----------
                    jlt              R4,1,@end_main
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain}
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
            
        
class T1Scan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float, 
                 length_stop: float, 
                 length_points: int, 
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpModel,
                 divisor_ns: int = 60000):
        self.divisor_ns = divisor_ns
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='T1',
                         x_label_plot='Wait Length', 
                         x_unit_plot='[ns].', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=length_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)

        
    def add_initvalues(self):
        start_ns = round(self.x_start * 1e9)
        initparameter = f"""
                    move             {start_ns},R4            
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += initparameter
        
        
    def add_mainpulse(self):
        """
        Because one 'wait' instruction can take no longer than 65534ns, we will divide it by 60us.
        
        Here I use R11 as a deepcopy of R4 and wait a few multiples of 60us first.
        After each wait we substract 60000 from R11.
        Then if R11 is smaller than 60us, we wait the remainder.
        """
        pi_pulse = {q: [f'X180_{ss}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        drive_length_ns = round(self.cfg.variables['common/qubit_pulse_length'] * 1e9)
        self.add_pulse(pi_pulse, drive_length_ns, 'T1PIpulse')
        
        step_ns = round(self.x_step * 1e9)
        main = f"""
                #-----------Main-----------
                    jlt              R4,1,@end_main
                    move             R4,R11            
                    
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
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpSinModel,
                 divisor_ns: int = 60000,
                 artificial_detuning: float = 0.0):
        self.divisor_ns = divisor_ns
        self.artificial_detuning = artificial_detuning
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='Ramsey',
                         x_label_plot='Wait Length', 
                         x_unit_plot='[ns].', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=length_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)

        
    def add_initvalues(self):
        """
        We will use R4 for wait time, R12 for angle of VZ gate.
        """
        start_ns = round(self.x_start * 1e9)
        start_ADphase = round(self.x_start * self.artificial_detuning * 1e9)  
        initparameter = f"""
                    move             {start_ns},R4            
                    move             {start_ADphase},R12
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += initparameter
        
        
    def add_mainpulse(self):
        """
        Because one 'wait' instruction can take no longer than 65534ns, we will divide it by 60us.
        
        Here, all register is 32bit, which can only store integer [-2e31,2e31).
        If we do Ramsey with more than 2 phase cycle, then it may cause error.
        Thus we substract 1e9 of R12 when it exceed 1e9, which is one phase cycle of Qblox.
        The wait trick is similar to T1Scan.
        """
        half_pi_pulse = {q: [f'X90_{ss}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        drive_length_ns = round(self.cfg.variables['common/qubit_pulse_length'] * 1e9)
        self.add_pulse(half_pi_pulse, drive_length_ns, 'RamseyHalfPIpulse')
        
        step_ns = round(self.x_step * 1e9)
        step_ADphase = round(self.x_step * self.artificial_detuning * 1e9)  
        # Qblox cut one phase circle into 1e9 pieces.
        main = f"""
                #-----------Main-----------
                    jlt              R4,1,@end_main
                    move             R4,R11            
                    
                    jlt              R12,1000000000,@mlt_wait
                    sub              R12,1000000000
                    
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
        
        self.add_pulse(half_pi_pulse, drive_length_ns, 'RamseyHalfPIpulse')
        
        
class EchoScan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float, 
                 length_stop: float, 
                 length_points: int, 
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpModel,
                 divisor_ns: int = 60000,
                 echo_type: str = 'CP'):
        self.divisor_ns = divisor_ns
        self.echo_type = echo_type
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='Echo',
                         x_label_plot='Wait Length', 
                         x_unit_plot='[ns].', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=length_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)

        
    def add_initvalues(self):
        start_half_ns = round(self.x_start / 2 * 1e9)
        initparameter = f"""
                    move             {start_half_ns},R4            
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += initparameter
        
        
    def add_mainpulse(self):
        """
        Here R4 only represent half of the total waiting time.
        """
        half_pi_pulse = {q: [f'X90_{ss}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        drive_length_ns = round(self.cfg.variables['common/qubit_pulse_length'] * 1e9)
        self.add_pulse(half_pi_pulse, drive_length_ns, 'Echo1stHalfPIpulse')
        
        step_half_ns = round(self.x_step / 2 * 1e9)
        main = f"""
                #-----------Main1-----------
                    jlt              R4,1,@end_main1
                    move             R4,R11            
                    
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
        self.add_pulse(pi_pulse, drive_length_ns, 'EchoPIpulse')
        
        main = f"""
                #-----------Main2-----------
                    jlt              R4,1,@end_main2
                    move             R4,R11            
                    
        mlt_wait2:  jlt              R11,{self.divisor_ns},@rmd_wait2
                    wait             {self.divisor_ns}
                    sub              R11,{self.divisor_ns},R11
                    jmp              @mlt_wait2
                    
        rmd_wait2:  wait             R11
                    
        end_main2:  add              R4,{step_half_ns},R4
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += main
        
        self.add_pulse(half_pi_pulse, drive_length_ns, 'Echo2ndHalfPIpulse')
        
        
class CalibrateClassification(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 level_start: int, 
                 level_stop: int,  
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 1000):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='CalibrateClassification',
                         x_label_plot='Level', 
                         x_unit_plot='', 
                         x_start=level_start, 
                         x_stop=level_stop, 
                         x_points=level_stop-level_start+1, 
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops)
    
    
    def add_initvalues(self):
        for qudit in self.qudits: self.sequences[qudit]['program'] += f"""
                    move             {self.x_start},R4            
        """
        
        
    def add_mainpulse(self):
        """
        Here we add all PI pulse to our sequence program based on level_stop.
        We will use R4 to represent level and jlt instruction to skip later PI pulse.
        """
        drive_length_ns = round(self.cfg.variables['common/qubit_pulse_length'] * 1e9)     
        
        for qudit in self.qudits:         
            main = """
                #-----------Main-----------
                    jlt              R4,1,@end_main
            """
            
            for level in range(self.x_stop):
                if qudit.startswith('Q'):
                    freq = round(self.cfg.variables[f'{qudit}/{level}{level+1}/mod_freq'] * 4)
                    gain = round(self.cfg.variables[f'{qudit}/{level}{level+1}/amp_180'] * 32768)
                    main += f"""
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain}
                    play             0,0,{drive_length_ns} 
                    jlt              R4,{level+2},@end_main
                    """
                elif qudit.startswith('R'):
                    main += f"""
                    wait             {drive_length_ns} 
                    jlt              R4,{level+2},@end_main
                    """
            
            main += """
        end_main:   add              R4,1,R4
            """
            self.sequences[qudit]['program'] += main
        