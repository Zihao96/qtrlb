from qtrl.calibration.calibration import Scan


class DriveAmplitudeScan(Scan):
    
    def __init__(self, 
                 cfg,  
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 amp_start: float | list, 
                 amp_stop: float | list, 
                 npoints: int, 
                 subspace: list = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 fitmodel = None,
                 error_amplification_factor: int = 1):
        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         x_name='drive_amplitude',
                         # x_label='drive_amplitude', 
                         # x_unit='a.u.', 
                         x_start=amp_start, 
                         x_stop=amp_stop, 
                         npoints=npoints, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         fitmodel=fitmodel)
        
        self.error_amplification_factor = error_amplification_factor

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



        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        