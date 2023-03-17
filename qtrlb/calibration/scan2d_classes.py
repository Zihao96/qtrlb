import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.offsetbox import AnchoredText
from qtrlb.calibration.calibration import Scan2D
from qtrlb.calibration.scan_classes import RabiScan, LevelScan
from qtrlb.processing.processing import gmm_fit, gmm_predict, normalize_population




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
                         x_plot_label='Pulse Length',
                         x_plot_unit='ns',
                         x_start=length_start,
                         x_stop=length_stop,
                         x_points=length_points,
                         y_plot_label='Frequency', 
                         y_plot_unit='MHz', 
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
    """ Sweep readout frequency with different qubit state.
        Calculate readout fidelity for each frequency and show the spectrum for all qubit state.
        Require calibrated PI pulse when scaning more than ground level.
        
        Note from Zihao(03/16/2023):
        I'm sorry this code is so long. This scan doesn't fit well into our calibration framework, \
        and we have to rebuild some wheel. Although it's possible to leave an interface in some \
        parent method, that will make things too thick, too ugly and hard to read.
    """
    def __init__(self,
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 level_start: int,
                 level_stop: int,
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
                         x_plot_label='Level',
                         x_plot_unit='arb',
                         x_start=level_start,
                         x_stop=level_stop,
                         x_points=level_stop-level_start+1,
                         y_plot_label='Frequency', 
                         y_plot_unit='kHz', 
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
        for r in self.readout_resonators: 
            assert all(self.cfg[f'variables.{r}/readout_levels'] == self.x_values), \
                    f'Please check readout levels of {r}!'
        

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
            
        
    def process_data(self, compensate_ED: bool = True):
        """
        Here we override the parent method since the processing for this Scan has no similarity to \
        other Scan. Code is similar to CalibrateClassification.plot_main()
        """
        shape = (2, self.n_reps, self.y_points, self.x_points)
        electrical_delay = self.cfg['variables.common/electrical_delay'] if compensate_ED else 0
        phase_offset = np.exp(1j * 2 * np.pi * self.y_values * electrical_delay)
        
        for r, data_dict in self.measurement.items():
            # First get averaged complex IQ vector for each frequency to plot spectrum.
            data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
            data_dict['IQaveraged_readout'] = np.mean(data_dict['Reshaped_readout'], axis=1)
            data_dict['IQcomplex_readout'] = (data_dict['IQaveraged_readout'][0]
                                             + 1j * data_dict['IQaveraged_readout'][1])
            data_dict['IQEDcompensated_readout'] = (data_dict['IQcomplex_readout'].T * phase_offset).T
            
            # Then loop all frequency and use GMM model to fit them and get confusion matrix.
            data_dict['GMMfitted'] = {}
            data_dict['to_fit'] = []
            
            for y in range(self.y_points):
                sub_dict = {}
                means = np.zeros((self.x_points, 2))
                covariances = np.zeros(self.x_points)
                
                for x in range(self.x_points):
                    data = data_dict['Reshaped_readout'][..., y, x]
                    mean, covariance = gmm_fit(data, n_components=1)
                    means[x] = mean[0]
                    covariances[x] = covariance[0]
                    # Because the default form is one more layer nested.
                    
                sub_dict['means'] = means
                sub_dict['covariances'] = covariances
                sub_dict['GMMpredicted'] = gmm_predict(data_dict['Reshaped_readout'][..., y, :], 
                                                       means=means, covariances=covariances)
                sub_dict['confusionmatrix'] = normalize_population(sub_dict['GMMpredicted'], 
                                                                   n_levels=self.x_points)
                sub_dict['ReadoutFidelity'] = np.sqrt(sub_dict['confusionmatrix'].diagonal()).mean()
                
                data_dict['GMMfitted'][y] = sub_dict
                data_dict['to_fit'].append(sub_dict['ReadoutFidelity'])
                
            data_dict['to_fit'] = np.array(data_dict['to_fit'])
            
            
    def fit_data(self):
        super().fit_data(x=self.y_values)  # We're actually using 1D fitting Scan.fit_data().

        
    def plot(self):
        self.plot_main()
        self.plot_spectrum()
        self.plot_IQ()
        

    def plot_main(self, text_loc: str = 'lower right'):
        """
        Plot readout fidelity along with fitting result as function of readout frequency.
        Code is similar to Scan.plot_main()
        """
        for i, r in enumerate(self.readout_resonators):
            title = f'{self.datetime_stamp}, {self.scan_name}, {r}'
            xlabel = self.y_plot_label + f'[{self.y_plot_unit}]'
            ylabel = 'Readout Fidelity [a.u.]'
            
            fig, ax = plt.subplots(1, 1, dpi=150)
            ax.plot(self.y_values / self.y_unit_value, self.measurement[r]['to_fit'], 'k.')
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
            
            if self.fit_result[r] is not None: 
                # Raise resolution of fit result for smooth plot.
                x = np.linspace(self.y_start, self.y_stop, self.y_points * 3)  
                y = self.fit_result[r].eval(x=x)
                ax.plot(x / self.y_unit_value, y, 'm-')
                
                # AnchoredText stolen from Ray's code.
                fit_text = '\n'.join([fr'{v.name} = {v.value:0.3g}$\pm${v.stderr:0.1g}' \
                                      for v in self.fit_result[r].params.values()])
                anchored_text = AnchoredText(fit_text, loc=text_loc, prop={'color':'m'})
                ax.add_artist(anchored_text)

            fig.savefig(os.path.join(self.data_path, f'{r}.png'))
        
        
    def plot_spectrum(self):
        """
        Plot the phase and Log-magnitude of the IQ data as readout frequency for all levels.
        """
        for r in self.readout_resonators:
            data = self.measurement[r]['IQEDcompensated_readout']
            
            title = f'{self.date}/{self.time}, {self.scan_name}, {r}'
            xlabel = self.y_plot_label + f'[{self.y_plot_unit}]'
            ylabel = ['IQ-phase [rad]', 'IQ-LogMag [a.u.]']
            
            fig, ax = plt.subplots(2, 1, figsize=(8,8), dpi=150)
            ax[0].set(xlabel=xlabel, ylabel=ylabel[0], title=title)
            ax[1].set(xlabel=xlabel, ylabel=ylabel[1])
            
            for level in self.x_values:
                ax[0].plot(self.y_values / self.y_unit_value, np.angle(data[:, level]), label=f'|{level}>')
                ax[1].plot(self.y_values / self.y_unit_value, np.absolute(data[:, level]), label=f'|{level}>')

            ax[0].legend()
            ax[1].legend()
            fig.savefig(os.path.join(self.data_path, f'{r}_spectrum.png'))
            
            
    def plot_IQ(self):
        """
        Plot IQ data for all frequency, each frequency will have a plot will all levels.
        Code is similar to Scan.plot_IQ()
        """
        for r in self.readout_resonators:
            Is, Qs = self.measurement[r]['Reshaped_readout']
            left, right = (np.min(Is), np.max(Is))
            bottom, top = (np.min(Qs), np.max(Qs))
            
            for y in range(self.y_points):                
                fig, ax = plt.subplots(1, self.x_points, figsize=(6 * self.x_points, 6), dpi=150)
                
                for x in range(self.x_points):
                    I = self.measurement[r]['Reshaped_readout'][0,:,y,x]
                    Q = self.measurement[r]['Reshaped_readout'][1,:,y,x]
                    c = self.measurement[r]['GMMfitted'][y]['GMMpredicted'][:,x]
                    cmap = LSC.from_list(None, plt.cm.tab10(self.cfg[f'variables.{r}/readout_levels']), 12)
                    
                    ax[x].scatter(I, Q, c=c, cmap=cmap, alpha=0.2)
                    ax[x].axvline(color='k', ls='dashed')    
                    ax[x].axhline(color='k', ls='dashed')
                    ax[x].set(xlabel='I', ylabel='Q', title=fr'${{\left|{self.x_values[x]}\right\rangle}}$', 
                              aspect='equal', xlim=(left, right), ylim=(bottom, top))
                    
                fig.savefig(os.path.join(self.data_path, f'{r}_IQplots', f'{y}.png'))
                plt.close(fig)
            
            
        
        