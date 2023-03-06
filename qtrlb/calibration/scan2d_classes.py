from lmfit import Model
from qtrlb.calibration.calibration import Scan2D
from qtrlb.calibration.scan_classes import RabiScan




class ChevronScan(Scan2D, RabiScan):
    def __init__(self,
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 detuning_start: float, 
                 detuning_stop: float, 
                 detuning_points: int, 
                 length_start: float = 0,
                 length_stop: float = 320e-9,
                 length_points: int = 81,
                 subspace: str | list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list = None,
                 fitmodel: Model = None,
                 init_waveform_idx: int = 11):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name='Chevron',
                         x_label_plot='Frequency', 
                         x_unit_plot='[GHz]', 
                         x_start=detuning_start, 
                         x_stop=detuning_stop, 
                         x_points=detuning_points, 
                         y_label_plot='Pulse Length',
                         y_unit_plot='[ns]',
                         y_start=length_start,
                         y_stop=length_stop,
                         y_points=length_points,
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        self.init_waveform_index = init_waveform_idx
        self.pulse_lengths = self.y_values
        assert self.x_stop >= self.x_start, 'Please use ascending value for detuning.'
        
        
    def add_xinit(self):
        """
        Here R4 is the detuning value, which need to be transformed into frequency of sequencer.
        """
        for i, qubit in enumerate(self.drive_qubits):
            ssb_freq_start = self.x_start + self.cfg[f'variables.{qubit}/{self.subspace[i]}/mod_freq']
            ssb_freq_start_4 = self.frequency_translator(ssb_freq_start)
            
            xinit = f"""
                    move             {ssb_freq_start_4},R4
            """
            self.sequences[qubit]['program'] += xinit
            
            
    def add_yinit(self):
        """
        Here R6 will be real pulse length, and R11 is the waveform index.
        So qubit play R11 for drive, resonator wait R6 for sync.
        """
        length_start_ns = round(self.y_start * 1e9)
        yinit = f"""
                    move             {length_start_ns},R6
                    move             {self.init_waveform_index},R11
        """
        for qudit in self.qudits: self.sequences[qudit]['program'] += yinit


    def add_mainpulse(self):
        """
        Qblox doesn't accept zero length waveform, so we use Label 'end_main' here.
        There is 4ns delay after each Rabi pulse before postpulse/readout.
        It's because the third index of 'play' instruction cannot be register.
        So we cannot set it as a variable, and wait will be separated.
        """
        length_step_ns = round(self.y_step * 1e9)
        
        for i, qubit in enumerate(self.drive_qubits):
            subspace = self.subspace[i]
            gain = round(self.cfg.variables[f'{qubit}/{subspace}/amp_rabi'] * 32768)
                    
            main = f"""
                 #-----------Main-----------
                    jlt              R6,1,@end_main
                    set_freq         R4
                    set_awg_gain     {gain},{gain}
                    play             R11,R11,4
                    wait             R6
        end_main:   add              R6,{length_step_ns},R6
                    add              R11,1,R11
            """  
            
            self.sequences[qubit]['program'] += main

        for resonator in self.readout_resonators:
            main = f"""
                 #-----------Main-----------
                    jlt              R6,1,@end_main
                    wait             4    
                    wait             R6
        end_main:   add              R6,{length_step_ns},R6
                    add              R11,1,R11
            """            
            self.sequences[resonator]['program'] += main


    def add_xvalue(self):
        
        ssb_freq_step_4 = round(self.x_step * 4)
        add_x = f"""
                    add              R4,{ssb_freq_step_4},R4
        """
        for q in self.drive_qubits:  self.sequences[q]['program'] += add_x
        
        
        
        
        
        