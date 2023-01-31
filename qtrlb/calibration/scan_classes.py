from qtrl.calibration.calibration import Scan


class DriveAmplitudeScan(Scan):
    
    def make_sequence(self):
        
        self.sequences = {}
        
        qubit_pulse_length = self.config.varman['common/qubit_pulse_length']
        resonator_pulse_length = self.config.varman['common/resonator_pulse_length']
        
        for q in self.drive_qubits:
            envelope = self.config.varman[f'Q{q}/pulse_shape']
            waveforms = {f'Q{q}': {'data': self.get_waveform(qubit_pulse_length, envelope), 
                                   'index': 0}}
            
            self.sequences[f'Q{q}'] = {'waveforms': waveforms,
                                       'weights': {},
                                       'acquisitions': {},
                                       'program':''}

            
        for r in self.readout_resonators:
            self.sequences[f'R{r}'] = {'waveforms': {},
                                       'weights': {},
                                       'acquisitions': {},
                                       'program':''}
        
        
        



        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        