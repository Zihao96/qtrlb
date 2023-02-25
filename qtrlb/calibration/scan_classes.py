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
                 x_points: int, 
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
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
                         x_points=x_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        
    def add_initparameter(self):
        for i, qubit in enumerate(self.drive_qubits):
            start = round(self.x_start * 32768)
            initparameter = f"""
                    move             {start},R0            
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
                    set_awg_gain     R0,R0
            """  
            
            for i in range(self.error_amplification_factor):
                main += f"""
                    play             0,0,{length}
                """
                
            main += f""" 
                    add              R0,{step},R0
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
                 length_stop: float = 360e-9, 
                 x_points: int = 91, 
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpSinModel):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='Rabi',
                         x_label_plot='Pulse Length', 
                         x_unit_plot='[ns].', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=x_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        
    def set_waveforms_acquisitions(self):
        """
        RabiScan need each element/x_point to have different pulse length 
        thus different waveforms for Qblox. We will generate them here.
        """
        self.check_waveform_length()
        super().set_waveforms_acquisitions()
        
        for q in self.drive_qubits:
            waveforms = {}
            for i in range(self.x_points):
                pulse_length_ns = round(self.x_values[i] * 1e9)
                if pulse_length_ns == 0: continue
            
                waveforms[f'{i}'] = {"data" : get_waveform(length=pulse_length_ns, 
                                                           shape=self.cfg.variables[f'{q}/pulse_shape']),
                                     "index": i}
            self.sequences[q]['waveforms'] = waveforms
        
        
    def check_waveform_length(self):
        """
        Check the total length of all waveforms since Qblox can only store 16384 samples.
        """
        total_length_ns = np.ceil(np.sum(self.x_values) * 1e9)
        assert total_length_ns < 16384, f'The total pulse length {total_length_ns}ns is too long! \n'\
            'Suggestion: np.linspace(0,200,101), np.linspace(0,320,81), np.linspace(0,360,91), '\
            'np.linspace(0,400,81), np.linspace(0,600,51)'
            

    def add_initparameter(self):
        """
        Here R0 will be real pulse length, and R1 is the waveform index.
        So qubit play R1 for drive, resonator wait R0 for sync.
        """
        start = round(self.x_start * 1e9)
        initparameter = f"""
                    move             {start},R0            
        """
        
        if start != 0: 
            for qudit in self.qudits: self.sequences[qudit]['program'] += initparameter
            
        else:  # Qblox doesn't accept zero length waveform (empty list).
            self.add_zero_start()
            if self.cfg.variables['common/heralding']: self.add_heralding()
            self.add_prepulse()
            self.add_postpulse()
            self.add_readout()
            self.add_zero_end()
            

    def add_zero_start(self):
        start = round(self.x_value[1] * 1e9)  # Set R0 (first pulse length) to second element.
        zero_start = f"""
                    move             {start},R0
                    wait_sync        8                               
                    reset_ph                                         
                    set_mrk          15                              
                    upd_param        8                               
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += zero_start
                
                
    def add_zero_end(self):
        zero_end = """        
                    add              R1,1,R1
                    set_mrk          0                               
                    upd_param        8                                
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += zero_end
        
        
    def add_mainpulse(self):
        """
        There is 4ns delay after each Rabi pulse before postpulse/readout.
        It's because the third index of 'play' instruction cannot be register.
        """
        step = round(self.x_step * 1e9)
        
        for i, qubit in enumerate(self.drive_qubits):
            subspace = self.subspace[i]
            freq = round(self.cfg.variables[f'{qubit}/{subspace}/mod_freq'] * 4) 
            gain = round(self.cfg.variables[f'{qubit}/{subspace}/amp_rabi'] * 32768)
                    
            main = f"""
                 #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain}
                    play             R1,R1,4
                    wait             R0
                    add              R0,{step},R0
            """  
            
            self.sequences[qubit]['program'] += main

        for resonator in self.readout_resonators:
            main = f"""
                 #-----------Main-----------
                    wait             4    
                    wait             R0
                    add              R0,{step},R0
            """            
            self.sequences[resonator]['program'] += main
            
        
class T1Scan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float, 
                 length_stop: float, 
                 x_points: int, 
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpModel,
                 divisor_ns: int = 60000):
        
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='T1',
                         x_label_plot='Wait Length', 
                         x_unit_plot='[ns].', 
                         x_start=length_start, 
                         x_stop=length_stop, 
                         x_points=x_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        self.divisor_ns = int(divisor_ns)
        
        
    def add_initparameter(self):
        """
        Here R0 will be real wait length? # TODO: Finish this one. Move zero_start to parent scan.
        """
        start = round(self.x_start * 1e9)
        initparameter = f"""
                    move             {start},R0            
        """
        
        if start != 0: 
            for qudit in self.qudits: self.sequences[qudit]['program'] += initparameter
            
        else:  # Qblox doesn't accept zero length waveform (empty list).
            self.add_zero_start()
            if self.cfg.variables['common/heralding']: self.add_heralding()
            self.add_prepulse()
            self.add_postpulse()
            self.add_readout()
            self.add_zero_end()
        
        
    def add_mainpulse(self):
        """
        Because one 'wait' instruction can take no longer than 65534ns,
        we will separate it into several instruction
        """
        pi_pulse = {q: [f'X180_{ss[0]}{ss[1]}'] for q, ss in zip(self.drive_qubits, self.subspace)}
        drive_length_ns = round(self.cfg.variables['common/qubit_pulse_length'] * 1e9)
        self.add_pulse(pi_pulse, drive_length_ns, 'T1PIpulse')
        
        for time in self.x_values:
            if time == 0: continue
            time_ns = round(time * 1e9)
            multiple = time_ns // self.divisor_ns
            remainder = time_ns % self.divisor_ns
            for i in range(multiple): self.add_wait(self.divisor_ns)

        
        
        
        
        
        
        
        
        
        
        