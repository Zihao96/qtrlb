import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.offsetbox import AnchoredText
import qtrlb.utils.units as u
from qtrlb.calibration.calibration import Scan2D
from qtrlb.calibration.scan_classes import RabiScan, LevelScan
from qtrlb.processing.fitting import QuadModel
from qtrlb.processing.processing import rotate_IQ, gmm_fit, gmm_predict, normalize_population, \
                                        get_readout_fidelity




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
                 pregate: dict = None,
                 postgate: dict = None,
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
                         pregate=pregate,
                         postgate=postgate,
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


    def add_main(self):
        """
        Qblox doesn't accept zero length waveform, so we use Label 'end_main' here.
        There is 4ns delay after each Rabi pulse before postgate/readout.
        It's because the third index of 'play' instruction cannot be register.
        So we cannot set it as a variable, and wait will be separated.
        """
        super().add_main(freq='R6')


    def add_yvalue(self):
        ssb_freq_step_4 = self.frequency_translator(self.y_step)
        for q in self.drive_qubits:  self.sequences[q]['program'] += f"""
                    add              R6,{ssb_freq_step_4},R6
        """
        
        
class ReadoutTemplateScan(Scan2D, LevelScan):
    """ Sweep a readout parameter (freq/amp/length) with different qubit state.
        Calculate readout fidelity for each parameter value and show the spectrum for all qubit state.
        Require calibrated PI gate when scaning more than ground level.
        This Scan has no __init__ and suppose to be a template inherited by child scan \
        and not to be called directly.
        Since the data in 'to_fit' is readout fidelity, we should always use level_to_fit at the lowest \
        readout level so that the index is always 0.
        
        Note from Zihao(03/16/2023):
        I'm sorry this code is so long. This type of scans don't fit well into our calibration framework, \
        and we have to rebuild some wheels. Although it's possible to leave an interface in some \
        parent method, that will make things too thick, too ugly and hard to read.
    """
    def check_attribute(self):
        super().check_attribute()
        assert not self.classification_enable, 'Please turn off classification.'
        
        
    def process_data(self, compensate_ED: bool = False):
        """
        Here we override the parent method since the processing for this Scan has no similarity to \
        other Scan. Code is similar to CalibrateClassification.fit_data()
        """
        shape = (2, self.n_reps, self.y_points, self.x_points)
        electrical_delay = self.cfg['variables.common/electrical_delay'] if compensate_ED else 0
        phase_offset = np.exp(1j * 2 * np.pi * self.y_values * electrical_delay)
        
        for r, data_dict in self.measurement.items():
            # First get averaged complex IQ vector for each y_value to plot spectrum.
            data_dict['Reshaped_readout'] = np.array(data_dict['Heterodyned_readout']).reshape(shape)
            data_dict['IQrotated_readout'] = rotate_IQ(data_dict['Reshaped_readout'], 
                                                       angle=self.cfg[f'process.{r}/IQ_rotation_angle'])
            data_dict['IQaveraged_readout'] = np.mean(data_dict['IQrotated_readout'], axis=1)
            data_dict['IQcomplex_readout'] = (data_dict['IQaveraged_readout'][0]
                                             + 1j * data_dict['IQaveraged_readout'][1])
            data_dict['IQEDcompensated_readout'] = (data_dict['IQcomplex_readout'].T * phase_offset).T
            
            # Then loop all y_values and use GMM model to fit them and get confusion matrix.
            data_dict['GMMfitted'] = {}
            data_dict['to_fit'] = []
            
            for y in range(self.y_points):
                sub_dict = {}
                means = np.zeros((self.x_points, 2))
                covariances = np.zeros(self.x_points)
                
                for x in range(self.x_points):
                    data = data_dict['IQrotated_readout'][..., y, x]
                    mean, covariance = gmm_fit(data, n_components=1)
                    means[x] = mean[0]
                    covariances[x] = covariance[0]
                    # Because the default form is one more layer nested.
                    
                sub_dict['means'] = means
                sub_dict['covariances'] = covariances
                sub_dict['GMMpredicted'] = gmm_predict(data_dict['IQrotated_readout'][..., y, :], 
                                                       means=means, covariances=covariances)
                sub_dict['confusionmatrix'] = normalize_population(sub_dict['GMMpredicted'], 
                                                                   n_levels=self.x_points)
                sub_dict['ReadoutFidelity'] = get_readout_fidelity(sub_dict['confusionmatrix'])
                
                data_dict['GMMfitted'][f'{y}'] = sub_dict
                data_dict['to_fit'].append(sub_dict['ReadoutFidelity'])
                
            data_dict['to_fit'] = np.array([data_dict['to_fit']])
            # Do not delete this nested structure.
            
            
    def fit_data(self):
        super().fit_data(x=self.y_values)  # We're actually using 1D fitting Scan.fit_data().

        
    def plot(self):
        self.plot_main()
        self.plot_spectrum()
        self.plot_IQ()
        

    def plot_main(self, text_loc: str = 'lower right'):
        """
        Plot readout fidelity along with fitting result as function of y_values.
        Code is similar to Scan.plot_main()
        """
        for i, r in enumerate(self.readout_resonators):
            title = f'{self.datetime_stamp}, {self.scan_name}, {r}'
            xlabel = self.y_plot_label + f'[{self.y_plot_unit}]'
            ylabel = 'Readout Fidelity [a.u.]'
            
            fig, ax = plt.subplots(1, 1, dpi=150)
            ax.plot(self.y_values / self.y_unit_value, self.measurement[r]['to_fit'][0], 'k.')
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
        Plot the phase and Log-magnitude of the IQ data as y_values for all levels.
        For readout amplitude and length, it may not be approprite to call it spectrum.
        But, you get what I mean here :)
        """
        for r in self.readout_resonators:
            data = self.measurement[r]['IQEDcompensated_readout']
            
            title = f'{self.datetime_stamp}, {self.scan_name}, {r}'
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
        Plot IQ data for all y_values, each y_value will have a plot with all levels.
        Code is similar to Scan.plot_IQ()
        """
        for r in self.readout_resonators:
            Is, Qs = self.measurement[r]['IQrotated_readout']
            left, right = (np.min(Is), np.max(Is))
            bottom, top = (np.min(Qs), np.max(Qs))
            
            for y in range(self.y_points):                
                fig, ax = plt.subplots(1, self.x_points, figsize=(6 * self.x_points, 6), dpi=150)
                
                for x in range(self.x_points):
                    I = self.measurement[r]['IQrotated_readout'][0,:,y,x]
                    Q = self.measurement[r]['IQrotated_readout'][1,:,y,x]
                    c = self.measurement[r]['GMMfitted'][f'{y}']['GMMpredicted'][:,x]
                    cmap = LSC.from_list(None, plt.cm.tab10(self.cfg[f'variables.{r}/readout_levels']), 12)
                    
                    ax[x].scatter(I, Q, c=c, cmap=cmap, alpha=0.2)
                    ax[x].axvline(color='k', ls='dashed')    
                    ax[x].axhline(color='k', ls='dashed')
                    ax[x].set(xlabel='I', ylabel='Q', title=fr'${{\left|{self.x_values[x]}\right\rangle}}$', 
                              aspect='equal', xlim=(left, right), ylim=(bottom, top))
                    
                fig.savefig(os.path.join(self.data_path, f'{r}_IQplots', f'{y}.png'))
                plt.close(fig)        
        
        
class ReadoutFrequencyScan(ReadoutTemplateScan):
    def __init__(self,
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 level_start: int,
                 level_stop: int,
                 detuning_start: float, 
                 detuning_stop: float, 
                 detuning_points: int, 
                 pregate: dict = None,
                 postgate: dict = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list = None,
                 fitmodel: Model = QuadModel):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name='ReadoutFrequency',
                         x_plot_label='Level',
                         x_plot_unit='arb',
                         x_start=level_start,
                         x_stop=level_stop,
                         x_points=level_stop-level_start+1,
                         y_plot_label='Readout Frequency', 
                         y_plot_unit='kHz', 
                         y_start=detuning_start, 
                         y_stop=detuning_stop, 
                         y_points=detuning_points, 
                         pregate=pregate,
                         postgate=postgate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)  
        

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
        Instead of using add_gate method, here we directly access the sequencer instruction.
        For more details, please check qtrlb.utils.pulses.
        """
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        readout_length_ns = round(self.cfg.variables['common/resonator_pulse_length'] * 1e9)
        length = tof_ns + readout_length_ns
        
        for qudit in self.qudits:
            if qudit.startswith('Q'):
                readout = f"""
                # -----------Readout-----------
                    wait             {length}
                """
            elif qudit.startswith('R'):
                gain = round(self.cfg.variables[f'{qudit}/amp'] * 32768)
                readout = f"""
                # -----------Readout-----------
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
        super().process_data(compensate_ED=True)


class ReadoutAmplitudeScan(ReadoutTemplateScan):
    def __init__(self,
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 level_start: int,
                 level_stop: int,
                 amp_start: float, 
                 amp_stop: float, 
                 amp_points: int, 
                 pregate: dict = None,
                 postgate: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = None):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name='ReadoutAmplitude',
                         x_plot_label='Level',
                         x_plot_unit='arb',
                         x_start=level_start,
                         x_stop=level_stop,
                         x_points=level_stop-level_start+1,
                         y_plot_label='Readout Amplitude', 
                         y_plot_unit='arb', 
                         y_start=amp_start, 
                         y_stop=amp_stop, 
                         y_points=amp_points, 
                         pregate=pregate,
                         postgate=postgate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)  
        

    def add_yinit(self):
        super().add_yinit()
        
        for r in self.readout_resonators:
            gain = round(self.y_start * 32768)
            yinit = f"""
                    move             {gain},R6
            """
            self.sequences[r]['program'] += yinit
        
        
    def add_readout(self):
        """
        Instead of using add_gate method, here we directly access the sequencer instruction.
        For more details, please check qtrlb.utils.pulses.
        """
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        readout_length_ns = round(self.cfg.variables['common/resonator_pulse_length'] * 1e9)
        length = tof_ns + readout_length_ns
        
        for qudit in self.qudits:
            if qudit.startswith('Q'):
                readout = f"""
                # -----------Readout-----------
                    wait             {length}
                """
            elif qudit.startswith('R'):
                freq = round(self.cfg[f'variables.{qudit}/mod_freq'] * 4)
                readout = f"""
                # -----------Readout-----------
                    set_freq         {freq}
                    set_awg_gain     R6,R6
                    play             0,0,{tof_ns} 
                    acquire          0,R1,{length - tof_ns}
                """

            self.sequences[qudit]['program'] += readout
        
        
    def add_yvalue(self):
        gain_step = round(self.y_step * 32768)
        for r in self.readout_resonators:  self.sequences[r]['program'] += f"""
                    add              R6,{gain_step},R6    """


class ReadoutLengthAmpScan(ReadoutAmplitudeScan):
    """ Run ReadoutAmplitudeScan with different readout and integration length.
        In principle it should be a 3D scan, but we can only change these integration length \
        in QCoDeS layer, so the easiest way is to make length at the outermost layer. 
        We achieve it by extending 'run' method.
        The plot and possible fit result will be saved in last experiment folder.
        
        Note from Zihao(04/04/2023):
        I do so since I believe the readout length and amplitude have a relatively wide range \
        so that all values within these range can give a reasonable result. 
        Even if noise change the optimal point, it will not be too much.
    """
    def __init__(self,
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 level_start: int,
                 level_stop: int,
                 amp_start: float, 
                 amp_stop: float, 
                 amp_points: int, 
                 length_start: float = 100 * u.ns,
                 length_stop: float = 5000 * u.ns,
                 length_points: int = 50,
                 pregate: dict = None,
                 postgate: dict = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list = None,
                 fitmodel: Model = None):

        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         level_start=level_start,
                         level_stop=level_stop,
                         amp_start=amp_start, 
                         amp_stop=amp_stop, 
                         amp_points=amp_points, 
                         pregate=pregate,
                         postgate=postgate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)

        self.length_start = length_start
        self.length_stop = length_stop
        self.length_points = length_points
        self.length_plot_label = 'Readout Length'
        self.length_plot_unit = 'ns'
        
        assert 16384 * u.ns > self.length_stop >= self.length_start > 0, \
            'Readout length must be ascending values in (0, 16384) ns.'
        self.length_values = np.linspace(self.length_start, self.length_stop, self.length_points)
        self.length_unit_value = getattr(u, self.length_plot_unit)


    def run(self, 
            experiment_suffix: str = '',
            n_pyloops: int = 1):
        
        for length in self.length_values:
            self.cfg['variables.common/resonator_pulse_length'] = length
            self.cfg['variables.common/integration_length'] = length
            
            super().run(experiment_suffix=f'{experiment_suffix}_{round(length*1e9)}ns',
                        n_pyloops=n_pyloops)
            
            plt.close('all')
                  
        # Load back the original yaml files.
        # Here you see the power of separating save and set, load and get. :)
        self.cfg.load()
        self.plot_full_result()
        
        
    def plot_full_result(self):
        """
        Combine all data with different length and make the 2D plot.
        The figure will be saved in the last experiment folder.
        """
        self.data_all_lengths = {}
        self.figures = {}
        
        for r in self.readout_resonators:
            self.data_all_lengths[r] = []
            
            for measurement in self.measurements:
                self.data_all_lengths[r].append(measurement[r]['to_fit'][0])
                
             
            title = f'{self.datetime_stamp}, Readout Length-Amp Scan, {r}'  
            xlabel = self.y_plot_label + f'[{self.y_plot_unit}]'
            ylabel = self.length_plot_label + f'[{self.length_plot_unit}]'
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
            
            image = ax.imshow(self.data_all_lengths[r], cmap='RdBu_r', interpolation='none', aspect='auto', 
                              origin='lower', extent=[np.min(self.y_values), np.max(self.y_values), 
                                                      np.min(self.length_values), np.max(self.length_values)])
            fig.colorbar(image, ax=ax, label='Fidelity', location='top')
            fig.savefig(os.path.join(self.data_path, f'{r}_full_result.png'))
            self.figures[r] = fig
