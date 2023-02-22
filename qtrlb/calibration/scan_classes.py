import numpy as np
from lmfit import Model
from qtrlb.calibration.calibration import Scan
from qtrlb.processing.fitting import SinModel, ExpSinModel


class DriveAmplitudeScan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 amp_start: float | list, 
                 amp_stop: float | list, 
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
            start = round(self.x_start[i] * 32768)
            initparameter = f"""
                    move             {start},R0            
            """
            self.sequences[qubit]['program'] += initparameter
            
            
    def add_mainpulse(self):
        length = round(self.cfg.variables['common/qubit_pulse_length'] * 1e9) 
        
        for i, qubit in enumerate(self.drive_qubits):
            step = round(self.x_step[i] * 32768)
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
            self.sequences[qubit]['program'] += main


class RabiScan(Scan):
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float | list, 
                 length_stop: float | list, 
                 x_points: int, 
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 level_to_fit: int | list = None,
                 fitmodel: Model = ExpSinModel):
        self.check_total_length()
        
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
        self.check_waveform_length()
        
        
    def check_waveform_length(self):
        """
        Check all qubits use same lengths so that they are properly synced.
        Check the total length of all waveforms since Qblox can only store 16384 samples.
        """
        # Ref: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
        assert [self.length_start[0]]*len(self.length_start) == self.length_start, \
                'All qubits should have length_start'
        assert [self.length_stop[0]]*len(self.length_stop) == self.length_stop, \
                'All qubits should have length_stop'
                
        total_length_ns = np.ceil(np.sum(self.x_values[0]) * 1e9)
        assert total_length_ns < 16384, f'The total pulse length {total_length_ns}ns is too long! \n'\
            'Suggestion: np.linspace(0,200,101), np.linspace(0,360,91), '\
            'np.linspace(0,400,81), np.linspace(0,600,51)'
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        