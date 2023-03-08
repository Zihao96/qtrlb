import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from qtrlb.calibration.calibration import Scan2D
from qtrlb.calibration.scan_classes import RabiScan, LevelScan




class ChevronScan(Scan2D, RabiScan):
    def __init__(self,
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 length_start: float,
                 length_stop: float,
                 length_points: int,
                 detuning_start: float, 
                 detuning_stop: float, 
                 detuning_points: int, 
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
                         x_label_plot='Pulse Length',
                         x_unit_plot='[ns]',
                         x_start=length_start,
                         x_stop=length_stop,
                         x_points=length_points,
                         y_label_plot='Frequency', 
                         y_unit_plot='[MHz]', 
                         y_start=detuning_start, 
                         y_stop=detuning_stop, 
                         y_points=detuning_points, 
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        self.init_waveform_index = init_waveform_idx
        
        
    def add_yinit(self):
        """
        Here R6 is the detuning value, which need to be transformed into frequency of sequencer.
        """
        super().add_yinit()
        
        for i, qubit in enumerate(self.drive_qubits):
            ssb_freq_start = self.y_start + self.cfg[f'variables.{qubit}/{self.subspace[i]}/mod_freq']
            ssb_freq_start_4 = self.frequency_translator(ssb_freq_start)
            
            yinit = f"""
                    move             {ssb_freq_start_4},R6
            """
            self.sequences[qubit]['program'] += yinit


    def add_mainpulse(self):
        """
        Qblox doesn't accept zero length waveform, so we use Label 'end_main' here.
        There is 4ns delay after each Rabi pulse before postpulse/readout.
        It's because the third index of 'play' instruction cannot be register.
        So we cannot set it as a variable, and wait will be separated.
        """
        super().add_mainpulse(freq='R6')


    def add_yvalue(self):
        ssb_freq_step_4 = self.frequency_translator(self.y_step)
        for q in self.drive_qubits:  self.sequences[q]['program'] += f"""
                    add              R6,{ssb_freq_step_4},R6
        """
        
        
class ReadoutFrequencyScan(Scan2D, LevelScan):
    def __init__(self,
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 level_start: float,
                 level_stop: float,
                 detuning_start: float, 
                 detuning_stop: float, 
                 detuning_points: int, 
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list = None,
                 fitmodel: Model = None):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name='ReadoutFrequency',
                         x_label_plot='Level',
                         x_unit_plot='',
                         x_start=level_start,
                         x_stop=level_stop,
                         x_points=level_stop-level_start+1,
                         y_label_plot='Frequency', 
                         y_unit_plot='[kHz]', 
                         y_start=detuning_start, 
                         y_stop=detuning_stop, 
                         y_points=detuning_points, 
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)        
        
        self.x_values = self.x_values.astype(int)
        assert not self.classification_enable, 'Please turn off classification.'
        

    def add_yinit(self):
        super().add_yinit()
        
        for r in self.readout_resonators:
            ssb_freq_start = self.y_start + self.cfg[f'variables.{r}/mod_freq']
            ssb_freq_start_4 = self.frequency_translator(ssb_freq_start)
            
            yinit = f"""
                    move             {ssb_freq_start_4},R6
            """
            self.sequences[r]['program'] += yinit
        
        
    def add_readout(self):
        """
        Instead of using add_pulse method, here we directly access the sequencer instruction.
        For more details, please check qtrlb.utils.pulses.
        """
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        readout_length_ns = round(self.cfg.variables['common/resonator_pulse_length'] * 1e9)
        length = tof_ns + readout_length_ns
        
        for qudit in self.qudits:
            if qudit.startswith('Q'):
                readout = f"""
                    wait             {length}
                """
            elif qudit.startswith('R'):
                gain = round(self.cfg.variables[f'{qudit}/amp'] * 32768)
                readout = f"""
                    set_freq         R6
                    set_awg_gain     {gain},{gain}
                    play             0,0,{tof_ns} 
                    acquire          0,R1,{length - tof_ns}
                """

            self.sequences[qudit]['program'] += readout
        
        
    def add_yvalue(self):
        ssb_freq_step_4 = self.frequency_translator(self.y_step)
        for r in self.readout_resonators:  self.sequences[r]['program'] += f"""
                    add              R6,{ssb_freq_step_4},R6    """
            
        
    def process_data(self):
        
        shape = (2, self.n_reps, self.y_points, self.x_points)
        electrical_delay = self.cfg['variables.common/electrical_delay']
        phase_offset = np.exp(1j * 2 * np.pi * self.y_values * electrical_delay)
        
        for r, data_dict in self.measurement.items():
            data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
            data_dict['IQaveraged_readout'] = np.mean(data_dict['Reshaped_readout'], axis=1)
            data_dict['IQcomplex_readout'] = (data_dict['IQaveraged_readout'][0]
                                             + 1j * data_dict['IQaveraged_readout'][1])
            data_dict['IQEDcompensated_readout'] = (data_dict['IQcomplex_readout'].T * phase_offset).T

        
    def plot(self):
        # TODO: allow it to do normal plot_main for 2D scan.
        # Also calculate and plot the readout fidelity.
        self.plot_spectrum()
        
        
    def plot_spectrum(self):
        """
        Plot the phase and Log-magnitude of the IQ data as readout frequency for all levels.
        """
        for r in self.readout_resonators:
            data = self.measurement[r]['IQEDcompensated_readout']
            
            title = f'{self.date}/{self.time}, {self.scan_name}, {r}'
            xlabel = self.y_label_plot + self.y_unit_plot
            ylabel = ['IQ-phase [rad]', 'IQ-LogMag [a.u.]']
            
            fig, ax = plt.subplots(2, 1, dpi=150)
            ax[0].set(xlabel=xlabel, ylabel=ylabel[0], title=title)
            ax[1].set(xlabel=xlabel, ylabel=ylabel[1], title=title)
            
            for level in self.x_values:
                ax[0].plot(self.y_values, np.angle(data[:, level]), label=f'|{level}>')
                ax[1].plot(self.y_values, np.absolute(data[:, level]), label=f'|{level}>')

            ax[0].legend()
            ax[1].legend()
            fig.savefig(os.path.join(self.data_path, f'{r}_spectrum.png'))
        
        
        