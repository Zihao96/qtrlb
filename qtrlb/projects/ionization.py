import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from abc import ABCMeta, abstractmethod
from matplotlib.offsetbox import AnchoredText

import qtrlb.utils.units as u
from qtrlb.config.config import MetaManager
from qtrlb.calibration.calibration import Scan, Scan2D
from qtrlb.calibration.scan_classes import Spectroscopy
from qtrlb.utils.waveforms import get_waveform
from qtrlb.utils.general_utils import make_it_list
from qtrlb.processing.fitting import fit, SpectroscopyModel




class IonizationBase(Scan, metaclass=ABCMeta):
    """ Base class for Ionization type of experiments.
        User must overload the __init__, add_xinit and add_main for child class.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def check_attribute(self):
        super().check_attribute()
        assert set(self.stimulation_tones).issubset(set(self.tones)), 'INZBase: stimulation_tones do not exist.'


    def make_tones_list(self):
        """
        Add stimulation tones to self.tones.
        The method here may change the order of self.tones, but not main_tones and readout_tones.
        """
        super().make_tones_list()
        self.tones += self.stimulation_tones
        self.tones = list(set(self.tones))

    
    @abstractmethod
    def add_xinit(self):
        super().add_xinit()

       
    @abstractmethod
    def add_main(self):
        super().add_main()
                 

class IonizationAmpScan(IonizationBase):
    """
    Sweep the amplitude of stimulation pulse in a typical ionization experiment:
    state preparation -> stimulation -> ringdown -> readout.
    This class allows arbitrary waveform and acquisition for stimulation tones.
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 amp_start: float,
                 amp_stop: float,
                 amp_points: int,
                 stimulation_tones: str | list[str],
                 stimulation_pulse_length: float,
                 ringdown_time: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None,
                 stimulation_waveform_idx: int = 1,
                 stimulation_acquisition_idx: int = 2):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='IonizationAmpScan',
                         x_plot_label='Stimulation Amplitude', 
                         x_plot_unit='arb', 
                         x_start=amp_start, 
                         x_stop=amp_stop, 
                         x_points=amp_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         stimulation_tones=make_it_list(stimulation_tones),
                         stimulation_pulse_length=stimulation_pulse_length,
                         ringdown_time=ringdown_time,
                         stimulation_waveform_idx=stimulation_waveform_idx,
                         stimulation_acquisition_idx=stimulation_acquisition_idx,
                         stimulation_pulse_length_ns=round(stimulation_pulse_length / u.ns),
                         ringdown_time_ns=round(ringdown_time / u.ns))
        

    def check_attribute(self):
        super().check_attribute()
        assert self.resonator_pulse_length_ns + self.stimulation_pulse_length_ns <= 16384, \
            f'Inz: The stimulation + readout pulse cannot exceed 16384ns.'


    def set_waveforms_acquisitions(self):
        """
        Add the simulation waveform to sequence_dict.

        Note from Zihao(2024/07/26):
        When stimulation tone is one of the readout tones, the update method of sequence_dict is required.
        When stimulation tone start with "R" but not actually readout tones, it still works properly.
        The only drawback is we might have useless bins under acquisitions['readout'].
        """
        super().set_waveforms_acquisitions(add_special_waveforms=False)

        for tone in self.stimulation_tones:
            waveforms = {'stimulation': {'data': get_waveform(length=self.stimulation_pulse_length_ns, 
                                                              shape=self.cfg[f'variables.{tone}/pulse_shape']), 
                                         'index': self.stimulation_waveform_idx}}
            self.sequences[tone]['waveforms'].update(waveforms)


    def add_xinit(self):
        """
        Here R4 is the amplitude of stimulation pulse.
        """
        super().add_xinit()
        
        for tone in self.stimulation_tones:
            x_start = self.gain_translator(self.x_start)
            self.sequences[tone]['program'] += f"""
                    move             {x_start},R4
            """


    def add_main(self):
        length = self.stimulation_pulse_length_ns + self.ringdown_time_ns
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        step = self.gain_translator(self.x_step)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)

                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     R4,R4
                    reset_ph
                    play             {self.stimulation_waveform_idx},{self.stimulation_waveform_idx},{tof_ns} 
                    acquire          {self.stimulation_acquisition_idx},R1,{length-tof_ns}
                    add              R4,{step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {length}
                """

            self.sequences[tone]['program'] += main


class IonizationAmpSquarePulse(IonizationBase):
    """
    Sweep the amplitude of stimulation pulse in a typical ionization experiment:
    state preparation -> stimulation -> ringdown -> readout.
    This class only allows square pulse for stimulation and does not allow acquisition.
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 amp_start: float,
                 amp_stop: float,
                 amp_points: int,
                 stimulation_tones: str | list[str],
                 stimulation_pulse_length: float,
                 ringdown_time: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='IonizationAmpSquarePulse',
                         x_plot_label='Stimulation Amplitude', 
                         x_plot_unit='arb', 
                         x_start=amp_start, 
                         x_stop=amp_stop, 
                         x_points=amp_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         stimulation_tones=make_it_list(stimulation_tones),
                         stimulation_pulse_length=stimulation_pulse_length,
                         ringdown_time=ringdown_time,
                         stimulation_pulse_length_ns=round(stimulation_pulse_length / u.ns),
                         ringdown_time_ns=round(ringdown_time / u.ns))


    def add_xinit(self):
        """
        Here R4 is the amplitude of stimulation pulse.
        """
        super().add_xinit()
        
        for tone in self.stimulation_tones:
            x_start = self.gain_translator(self.x_start)
            self.sequences[tone]['program'] += f"""
                    move             {x_start},R4
            """


    def add_main(self):
        step = self.gain_translator(self.x_step)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)

                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_offs     R4,R4
                    reset_ph
                    upd_param        {self.stimulation_pulse_length_ns} 
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                    add              R4,{step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {self.stimulation_pulse_length_ns + self.ringdown_time_ns}
                """

            self.sequences[tone]['program'] += main


class IonizationRingDownScan(IonizationBase):
    """
    Sweep the ring-down time in a typical ionization experiment:
    state preparation -> stimulation -> ringdown -> readout.
    This class only allows square pulse for stimulation and does not allow acquisition.
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 ringdown_start: float,
                 ringdown_stop: float,
                 ringdown_points: int,
                 stimulation_tones: str | list[str],
                 stimulation_pulse_length: float,
                 stimulation_amp: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None):

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='IonizationRingDownScan',
                         x_plot_label='Ringdown Time', 
                         x_plot_unit='us', 
                         x_start=ringdown_start, 
                         x_stop=ringdown_stop, 
                         x_points=ringdown_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         stimulation_tones=make_it_list(stimulation_tones),
                         stimulation_pulse_length=stimulation_pulse_length,
                         stimulation_amp=stimulation_amp,
                         stimulation_pulse_length_ns=round(stimulation_pulse_length / u.ns))


    def check_attribute(self):
        super().check_attribute()
        assert 8 * u.ns < self.x_start < self.x_stop < 65536 * u.ns, \
            'IRD: All ringdown time must be in range (8, 65536) ns.'


    def add_xinit(self):
        """
        Here R4 is the ringdown time in (ns).
        """
        super().add_xinit()
        for tone in self.tones:
            self.sequences[tone]['program'] += f"""
                    move             {round(self.x_start * 1e9)},R4
            """


    def add_main(self):
        amp = round(self.stimulation_amp * 32768)
        step_ns = round(self.x_step * 1e9)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)
                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_offs     {amp},{amp}
                    reset_ph
                    upd_param        {self.stimulation_pulse_length_ns}
                    set_awg_offs     0,0
                    sub              R4,8,R11
                    upd_param        8
                    wait             R11
                    add              R4,{step_ns},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {self.stimulation_pulse_length_ns}
                    wait             R4
                    add              R4,{step_ns},R4
                """
            self.sequences[tone]['program'] += main


class IonizationLengthAmpScan(Scan2D, IonizationBase):
    """
    Sweep the stimulation amplitude (x) and length (y) in a typical ionization experiment:
    state preparation -> stimulation -> ringdown -> readout.
    This class only allows square pulse for stimulation and does not allow acquisition.
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 amp_start: float,
                 amp_stop: float,
                 amp_points: int,
                 length_start: float, 
                 length_stop: float, 
                 length_points: int,
                 stimulation_tones: str | list[str],
                 ringdown_time: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='IonizationLengthAmpScan',
                         x_plot_label='Stimulation Amplitude',
                         x_plot_unit='arb',
                         x_start=amp_start,
                         x_stop=amp_stop,
                         x_points=amp_points,
                         y_plot_label='Stimulation Length', 
                         y_plot_unit='ns', 
                         y_start=length_start, 
                         y_stop=length_stop, 
                         y_points=length_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         stimulation_tones=make_it_list(stimulation_tones),
                         ringdown_time=ringdown_time,
                         ringdown_time_ns=round(ringdown_time / u.ns))
        

    def check_attribute(self):
        super().check_attribute()        
        assert 8 * u.ns < self.y_start < self.y_stop < 65536 * u.ns, \
            'ILAS: All stimulation length must be in range (8, 65536) ns.'


    def add_yinit(self):
        """
        Here R6 is the length of stimulation pulse.
        """
        super().add_yinit()
        for tone in self.tones:
            self.sequences[tone]['program'] += f"""
                    move             {round(self.y_start / u.ns)},R6
            """


    def add_main(self):
        x_step = self.gain_translator(self.x_step)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)

                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_offs     R4,R4
                    reset_ph
                    sub              R6,8,R11
                    upd_param        8
                    wait             R11
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                    add              R4,{x_step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             R6
                    wait             {self.ringdown_time_ns}
                    add              R4,{x_step},R4
                """

            self.sequences[tone]['program'] += main


    def add_yvalue(self):
        for tone in self.tones:  
            self.sequences[tone]['program'] += f"""
                    add              R6,{round(self.y_step / u.ns)},R6
            """


class IonizationAmpSpectroscopy(Scan2D, IonizationBase, Spectroscopy):
    """
    Sweep the spectroscopy frequency (y) and stimulation amplitude (x) in a typical ionization experiment:
    state preparation -> stimulation -> ringdown -> readout.
    The spectroscopy pulse is added before the end of the simulation.
    This class only allows square pulse for stimulation and does not allow acquisition.
    Ref: https://doi.org/10.1103/PhysRevLett.117.190503
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 detuning_start: float, 
                 detuning_stop: float, 
                 detuning_points: int,
                 amp_start: float,
                 amp_stop: float,
                 amp_points: int,
                 stimulation_tones: str | list[str],
                 stimulation_pulse_length: float,
                 ringdown_time: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = SpectroscopyModel):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='IonizationAmpSpectroscopy',
                         x_plot_label='Pulse Detuning',
                         x_plot_unit='MHz',
                         x_start=detuning_start,
                         x_stop=detuning_stop,
                         x_points=detuning_points,
                         y_plot_label='Stimulation Amplitude', 
                         y_plot_unit='arb', 
                         y_start=amp_start, 
                         y_stop=amp_stop, 
                         y_points=amp_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         stimulation_tones=make_it_list(stimulation_tones),
                         stimulation_pulse_length=stimulation_pulse_length,
                         ringdown_time=ringdown_time,
                         stimulation_pulse_length_ns=round(stimulation_pulse_length / u.ns),
                         ringdown_time_ns=round(ringdown_time / u.ns))


    def add_xinit(self):
        """
        We will call Spectroscopy.add_xinit() here.
        """
        super().add_xinit()

        
    def add_yinit(self):
        """
        Here R6 is the amplitude of the stimulation pulse.
        """
        super().add_yinit()
        for tone in self.stimulation_tones:
            y_start = self.gain_translator(self.y_start)
            self.sequences[tone]['program'] += f"""
                    move             {y_start},R6
            """


    def add_main(self):            
        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)
                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_offs     R6,R6
                    reset_ph
                    upd_param        {self.stimulation_pulse_length_ns} 
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                """

            elif tone in self.main_tones:
                step = self.frequency_translator(self.x_step)
                gain = round(self.cfg.variables[f'{tone}']['amp_180'] * 32768)
                gain_drag = round(gain * self.cfg.variables[f'{tone}']['DRAG_weight'])
                main = f"""
                #-----------Main-----------
                    wait             {self.stimulation_pulse_length_ns - self.qubit_pulse_length_ns}
                    set_freq         R4
                    set_awg_gain     {gain},{gain_drag}
                    play             0,1,{self.qubit_pulse_length_ns + self.ringdown_time_ns}
                    add              R4,{step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {self.stimulation_pulse_length_ns + self.ringdown_time_ns}
                """

            self.sequences[tone]['program'] += main


    def add_yvalue(self):
        y_step = self.gain_translator(self.y_step)
        for tone in self.stimulation_tones:  
            self.sequences[tone]['program'] += f"""
                    add              R6,{y_step},R6
            """


    def fit_data(self, x: list | np.ndarray = None, **fitting_kwargs):
        """
        We won't fit 2D data, instead, we treat each amp as an independent spectroscopy and fit it.
        See Scan.fit_data() as reference.
        """
        self.fit_result = {rr: [] for rr in self.readout_resonators}
        if self.fitmodel is None: return
        if x is None: x = self.x_values
        
        for i, rr in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']

            for j in range(self.y_points):
                try:
                    result = fit(input_data=self.measurement[rr]['to_fit'][level_index][j],
                                 x=x, fitmodel=self.fitmodel, t=self.qubit_pulse_length_ns * u.ns,
                                 **fitting_kwargs)
                    self.fit_result[rr].append(result)
                    
                    params = {v.name:{'value':v.value, 'stderr':v.stderr} for v in result.params.values()}
                    self.measurement[rr][f'fit_result_{j}'] = params
                    self.measurement[rr]['fit_model'] = str(result.model)
                except Exception:
                    self.fitting_traceback = traceback.format_exc()  # Return a string to debug.
                    print(f'IAS: Failed to fit {rr} {j}-th amp data. ')
                    self.measurement[rr][f'fit_result_{j}'] = None
                    self.measurement[rr]['fit_model'] = str(self.fitmodel)
    

    def plot_main(self, text_loc: str = 'lower right', dpi: int = 150):
        """
        Here we will save all the plot without showing in console or make them attributes.
        See Scan.plot_main() as reference.
        """
        for i, rr in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']      

            if self.classification_enable:
                ylabel = fr'$P_{{\left|{self.level_to_fit[i]}\right\rangle}}$'
            else:
                ylabel = 'I-Q Coordinate (Rotated) [a.u.]'

            for j, amp in enumerate(self.y_values):
                fig, ax = plt.subplots(1, 1, dpi=dpi)
                ax.plot(self.x_values / self.x_unit_value, self.measurement[rr]['to_fit'][level_index][j], 'k.')
                ax.set(xlabel=self.x_plot_label + f'[{self.x_plot_unit}]', ylabel=ylabel, 
                       title=f'{self.datetime_stamp}, {self.scan_name}, {rr}, Amp{amp}')

                if self.measurement[rr][f'fit_result_{j}'] is not None: 
                    # Raise resolution of fit result for smooth plot.
                    x = np.linspace(self.x_start, self.x_stop, self.x_points * 3)  
                    y = self.fit_result[rr][j].eval(x=x)
                    ax.plot(x / self.x_unit_value, y, 'm-')
                    
                    fit_text = '\n'.join([f'{v.name} = {v.value:0.3g}' for v in self.fit_result[rr][j].params.values()])
                    ax.add_artist(AnchoredText(fit_text, loc=text_loc, prop={'color':'m'}))

                fig.savefig(os.path.join(self.data_path, f'{rr}_amp{j}.png'))
                plt.close('all')

    
    def plot_populations(self, dpi: int = 150):
        """
        Here we will save all the plot without showing in console or make them attributes.
        See Scan.plot_population() as reference.
        """
        for rr in self.readout_resonators:
            for j, amp in enumerate(self.y_values):
                fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=dpi)
                for i, level in enumerate(self.cfg[f'variables.{rr}/readout_levels']):
                    ax[0].plot(self.x_values / self.x_unit_value, self.measurement[rr]['PopulationNormalized_readout'][i][j], 
                               c=self.color_list[level], ls='-', marker='.', label=fr'$P_{{{level}}}$')
                    ax[1].plot(self.x_values / self.x_unit_value, self.measurement[rr]['PopulationCorrected_readout'][i][j], 
                               c=self.color_list[level], ls='-', marker='.', label=fr'$P_{{{level}}}$')

                xlabel = f'{self.x_plot_label}[{self.x_plot_unit}]'
                ax[0].set(xlabel=xlabel, ylabel='Uncorrected populations', ylim=(-0.05, 1.05))
                ax[1].set(xlabel=xlabel, ylabel='Corrected populations', ylim=(-0.05, 1.05))
                ax[0].legend()
                ax[1].legend()
                ax[0].set_title(f'{self.datetime_stamp}, {self.scan_name}, {rr}, Amp{amp}')
                fig.savefig(os.path.join(self.data_path, f'{rr}_Population_Amp{j}.png'))
                fig.clear()
                plt.close('all')


class IonizationDelaySpectroscopy(Scan2D, IonizationBase, Spectroscopy):
    """
    Sweep the spectroscopy frequency (x) and delay time (y) in a typical ionization experiment:
    state preparation -> stimulation -> ringdown -> readout.
    The spectroscopy pulse is added during stimulation and ringdown.
    This class only allows square pulse for stimulation and does not allow acquisition.
    Ref: https://doi.org/10.1103/PhysRevLett.117.190503
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 detuning_start: float, 
                 detuning_stop: float, 
                 detuning_points: int,
                 time_start: float,
                 time_stop: float,
                 time_points: int,
                 stimulation_tones: str | list[str],
                 stimulation_pulse_length: float,
                 ringdown_time: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = SpectroscopyModel):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='IonizationDelaySpectroscopy',
                         x_plot_label='Pulse Detuning',
                         x_plot_unit='MHz',
                         x_start=detuning_start,
                         x_stop=detuning_stop,
                         x_points=detuning_points,
                         y_plot_label='Time', 
                         y_plot_unit='us', 
                         y_start=time_start, 
                         y_stop=time_stop, 
                         y_points=time_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         stimulation_tones=make_it_list(stimulation_tones),
                         stimulation_pulse_length=stimulation_pulse_length,
                         ringdown_time=ringdown_time,
                         stimulation_pulse_length_ns=round(stimulation_pulse_length / u.ns),
                         ringdown_time_ns=round(ringdown_time / u.ns))


    def check_attribute(self):
        super().check_attribute()
        assert 0 <= self.y_start < self.y_stop < self.stimulation_pulse_length_ns + self.ringdown_time_ns, \
            "IDS: All delay time must be in range [0, stimulation_pulse_length + ringdown_time) ns."
    

    def add_xinit(self):
        """
        We will call Spectroscopy.add_xinit() here.
        """
        super().add_xinit()

        
    def add_yinit(self):
        """
        Here R6 is the beginning time of the spectroscopy pulse.
        R6 = 0 means the spectroscopy pulse starts at the time as the stimulation.
        """
        super().add_yinit()
        for tone in self.main_tones:
            y_start_ns = round(self.y_start / u.ns)
            self.sequences[tone]['program'] += f"""
                    move             {y_start_ns},R6
            """


    def add_main(self):
        """
        Here R11 is the total wait time excluding the length of the spectroscopy pulse.
        R12 is the wait time after the end of the spectroscopy pulse. R12 = R11 - R6.
        So the sequence is R6--Spectroscopy--R12.
        """        
        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)
                amp = round(self.cfg.variables[f'{tone}/amp'] * 32768)
                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_offs     {amp},{amp}
                    reset_ph
                    upd_param        {self.stimulation_pulse_length_ns} 
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                """

            elif tone in self.main_tones:
                total_wait_time = self.stimulation_pulse_length_ns + self.ringdown_time_ns - self.qubit_pulse_length_ns
                gain = round(self.cfg.variables[f'{tone}']['amp_180'] * 32768)
                gain_drag = round(gain * self.cfg.variables[f'{tone}']['DRAG_weight'])
                step = self.frequency_translator(self.x_step)
                main = f"""
                #-----------Main-----------
                    move             {total_wait_time},R11
                    nop
                    sub              R11,R6,R12
                    jlt              R6,1,@spec_pls
                    wait             R6
        spec_pls:   set_freq         R4
                    set_awg_gain     {gain},{gain_drag}
                    play             0,1,{self.qubit_pulse_length_ns}
                    wait             R12
                    add              R4,{step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {self.stimulation_pulse_length_ns + self.ringdown_time_ns}
                """

            self.sequences[tone]['program'] += main


    def add_yvalue(self):
        y_step_ns = round(self.y_step / u.ns)
        for tone in self.main_tones:  
            self.sequences[tone]['program'] += f"""
                    add              R6,{y_step_ns},R6
            """


    def fit_data(self, x: list | np.ndarray = None, **fitting_kwargs):
        """
        We won't fit 2D data, instead, we treat each amp as an independent spectroscopy and fit it.
        See Scan.fit_data() as reference.
        """
        self.fit_result = {rr: [] for rr in self.readout_resonators}
        if self.fitmodel is None: return
        if x is None: x = self.x_values
        
        for i, rr in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']

            for j in range(self.y_points):
                try:
                    result = fit(input_data=self.measurement[rr]['to_fit'][level_index][j],
                                 x=x, fitmodel=self.fitmodel, t=self.qubit_pulse_length_ns * u.ns,
                                 **fitting_kwargs)
                    self.fit_result[rr].append(result)
                    
                    params = {v.name:{'value':v.value, 'stderr':v.stderr} for v in result.params.values()}
                    self.measurement[rr][f'fit_result_{j}'] = params
                    self.measurement[rr]['fit_model'] = str(result.model)
                except Exception:
                    self.fitting_traceback = traceback.format_exc()  # Return a string to debug.
                    print(f'IDS: Failed to fit {rr} {j}-th delay time data. ')
                    self.measurement[rr][f'fit_result_{j}'] = None
                    self.measurement[rr]['fit_model'] = str(self.fitmodel)


    def plot_main(self, text_loc: str = 'lower right', dpi: int = 150):
        """
        Here we will save all the plot without showing in console or make them attributes.
        See Scan.plot_main() as reference.
        """
        for i, rr in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']      

            if self.classification_enable:
                ylabel = fr'$P_{{\left|{self.level_to_fit[i]}\right\rangle}}$'
            else:
                ylabel = 'I-Q Coordinate (Rotated) [a.u.]'

            for j, time in enumerate(self.y_values):
                fig, ax = plt.subplots(1, 1, dpi=dpi)
                ax.plot(self.x_values / self.x_unit_value, self.measurement[rr]['to_fit'][level_index][j], 'k.')
                ax.set(xlabel=self.x_plot_label + f'[{self.x_plot_unit}]', ylabel=ylabel, 
                       title=f'{self.datetime_stamp}, {self.scan_name}, {rr}, Time{round(time / self.y_unit_value)}({self.y_plot_unit})')

                if self.measurement[rr][f'fit_result_{j}'] is not None: 
                    # Raise resolution of fit result for smooth plot.
                    x = np.linspace(self.x_start, self.x_stop, self.x_points * 3)  
                    y = self.fit_result[rr][j].eval(x=x)
                    ax.plot(x / self.x_unit_value, y, 'm-')
                    
                    fit_text = '\n'.join([f'{v.name} = {v.value:0.3g}' for v in self.fit_result[rr][j].params.values()])
                    ax.add_artist(AnchoredText(fit_text, loc=text_loc, prop={'color':'m'}))

                fig.savefig(os.path.join(self.data_path, f'{rr}_time{j}.png'))
                plt.close('all')

    
    def plot_populations(self, dpi: int = 150):
        """
        Here we will save all the plot without showing in console or make them attributes.
        See Scan.plot_population() as reference.
        """
        for rr in self.readout_resonators:
            for j, amp in enumerate(self.y_values):
                fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=dpi)
                for i, level in enumerate(self.cfg[f'variables.{rr}/readout_levels']):
                    ax[0].plot(self.x_values / self.x_unit_value, self.measurement[rr]['PopulationNormalized_readout'][i][j], 
                               c=self.color_list[level], ls='-', marker='.', label=fr'$P_{{{level}}}$')
                    ax[1].plot(self.x_values / self.x_unit_value, self.measurement[rr]['PopulationCorrected_readout'][i][j], 
                               c=self.color_list[level], ls='-', marker='.', label=fr'$P_{{{level}}}$')

                xlabel = f'{self.x_plot_label}[{self.x_plot_unit}]'
                ax[0].set(xlabel=xlabel, ylabel='Uncorrected populations', ylim=(-0.05, 1.05))
                ax[1].set(xlabel=xlabel, ylabel='Corrected populations', ylim=(-0.05, 1.05))
                ax[0].legend()
                ax[1].legend()
                ax[0].set_title(f'{self.datetime_stamp}, {self.scan_name}, {rr}, Amp{amp}')
                fig.savefig(os.path.join(self.data_path, f'{rr}_Population_Amp{j}.png'))
                fig.clear()
                plt.close('all')


################################################################################
class Ionization(Scan):
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 amp_start: float,
                 amp_stop: float,
                 amp_points: int,
                 stimulation_tones: str | list[str],
                 stimulation_pulse_length: float,
                 ringdown_time: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None,
                 stimulation_waveform_idx: int = 1,
                 stimulation_acquisition_idx: int = 2):

        self.stimulation_tones = make_it_list(stimulation_tones)

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='Ionization',
                         x_plot_label='Stimulation Amplitude', 
                         x_plot_unit='arb', 
                         x_start=amp_start, 
                         x_stop=amp_stop, 
                         x_points=amp_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)

        self.stimulation_pulse_length = stimulation_pulse_length
        self.ringdown_time = ringdown_time
        self.stimulation_waveform_idx = stimulation_waveform_idx
        self.stimulation_acquisition_idx = stimulation_acquisition_idx

        self.stimulation_pulse_length_ns = round(stimulation_pulse_length / u.ns)
        self.ringdown_time_ns = round(ringdown_time / u.ns)

        assert set(self.stimulation_tones).issubset(set(self.tones)), 'Inz: stimulation_tones do not exist.'
        assert self.resonator_pulse_length_ns + self.stimulation_pulse_length_ns <= 16384, \
            f'Inz: The stimulation + readout pulse cannot exceed 16384ns.'


    def make_tones_list(self):
        """
        Add stimulation tones to self.tones.
        The method here may change the order of self.tones, but not main_tones and readout_tones.
        """
        super().make_tones_list()
        self.tones += self.stimulation_tones
        self.tones = list(set(self.tones))


    def set_waveforms_acquisitions(self):
        """
        Add the simulation waveform to sequence_dict.

        Note from Zihao(2024/07/26):
        When stimulation tone is one of the readout tones, the update method of sequence_dict is required.
        When stimulation tone start with "R" but not actually readout tones, it still works properly.
        The only drawback is we might have useless bins under acquisitions['readout'].
        """
        super().set_waveforms_acquisitions(add_special_waveforms=False)

        for tone in self.stimulation_tones:
            waveforms = {'stimulation': {'data': get_waveform(length=self.stimulation_pulse_length_ns, 
                                                              shape=self.cfg[f'variables.{tone}/pulse_shape']), 
                                         'index': self.stimulation_waveform_idx}}
            acquisitions = {'stimulation': {'num_bins': self.num_bins, 'index': self.stimulation_acquisition_idx}}
            self.sequences[tone]['waveforms'].update(waveforms)
            self.sequences[tone]['acquisitions'].update(acquisitions)


    def add_xinit(self):
        """
        Here R4 is the amplitude of stimulation pulse.
        """
        super().add_xinit()
        
        for tone in self.stimulation_tones:
            x_start = self.gain_translator(self.x_start)
            self.sequences[tone]['program'] += f"""
                    move             {x_start},R4
            """


    def add_main(self):
        length = self.stimulation_pulse_length_ns + self.ringdown_time_ns
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        step = self.gain_translator(self.x_step)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)

                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     R4,R4
                    reset_ph
                    play             {self.stimulation_waveform_idx},{self.stimulation_waveform_idx},{tof_ns} 
                    acquire          {self.stimulation_acquisition_idx},R1,{length-tof_ns}
                    add              R4,{step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {length}
                """

            self.sequences[tone]['program'] += main


class IonizationRingDown(Scan):
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 ringdown_start: float,
                 ringdown_stop: float,
                 ringdown_points: int,
                 stimulation_tones: str | list[str],
                 stimulation_pulse_length: float,
                 stimulation_amp: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 1000,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None,
                 stimulation_waveform_idx: int = 1,
                 stimulation_acquisition_idx: int = 2):

        self.stimulation_tones = make_it_list(stimulation_tones)

        super().__init__(cfg=cfg,
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='IonizationRingDown',
                         x_plot_label='Ringdown Time', 
                         x_plot_unit='us', 
                         x_start=ringdown_start, 
                         x_stop=ringdown_stop, 
                         x_points=ringdown_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)

        self.stimulation_pulse_length = stimulation_pulse_length
        self.stimulation_amp = stimulation_amp
        self.stimulation_waveform_idx = stimulation_waveform_idx
        self.stimulation_acquisition_idx = stimulation_acquisition_idx
        self.stimulation_pulse_length_ns = round(stimulation_pulse_length / u.ns)

        assert self.x_start > 0 and self.x_stop < 65536 * u.ns, \
            'IRD: All ringdown time must be in range (0, 65536) ns.'
        assert set(self.stimulation_tones).issubset(set(self.tones)), 'IRD: stimulation_tones do not exist.'
        assert self.resonator_pulse_length_ns + self.stimulation_pulse_length_ns <= 16384, \
            'IRD: The stimulation + readout pulse cannot exceed 16384ns.'


    def make_tones_list(self):
        """
        Add stimulation tones to self.tones.
        The method here may change the order of self.tones, but not main_tones and readout_tones.
        """
        super().make_tones_list()
        self.tones += self.stimulation_tones
        self.tones = list(set(self.tones))


    def set_waveforms_acquisitions(self):
        """
        Add the simulation waveform to sequence_dict.

        Note from Zihao(2024/07/26):
        When stimulation tone is one of the readout tones, the update method of sequence_dict is required.
        When stimulation tone start with "R" but not actually readout tones, it still works properly.
        The only drawback is we might have useless bins under acquisitions['readout'].
        """
        super().set_waveforms_acquisitions(add_special_waveforms=False)

        for tone in self.stimulation_tones:
            waveforms = {'stimulation': {'data': get_waveform(length=self.stimulation_pulse_length_ns, 
                                                              shape=self.cfg[f'variables.{tone}/pulse_shape']), 
                                         'index': self.stimulation_waveform_idx}}
            acquisitions = {'stimulation': {'num_bins': self.num_bins, 'index': self.stimulation_acquisition_idx}}
            self.sequences[tone]['waveforms'].update(waveforms)
            self.sequences[tone]['acquisitions'].update(acquisitions)


    def add_xinit(self):
        """
        Here R4 is the amplitude of stimulation pulse.
        """
        super().add_xinit()
                
        start_ns = round(self.x_start * 1e9)
        for tone in self.tones:
            self.sequences[tone]['program'] += f"""
                    move             {start_ns},R4
            """


    def add_main(self):
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        stimulation_length = self.stimulation_pulse_length_ns
        amp = round(self.stimulation_amp * 32768)
        step_ns = round(self.x_step * 1e9)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)
                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     {amp},{amp}
                    reset_ph
                    play             {self.stimulation_waveform_idx},{self.stimulation_waveform_idx},{tof_ns} 
                    acquire          {self.stimulation_acquisition_idx},R1,{stimulation_length-tof_ns}
                    wait             R4
                    add              R4,{step_ns},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {stimulation_length}
                    wait             R4
                    add              R4,{step_ns},R4
                """
            self.sequences[tone]['program'] += main


class ACStarkSpectroscopy(Scan2D, Ionization, Spectroscopy):
    """
    Ref: https://doi.org/10.1103/PhysRevLett.117.190503
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 amp_start: float,
                 amp_stop: float,
                 amp_points: int,
                 detuning_start: float, 
                 detuning_stop: float, 
                 detuning_points: int,
                 stimulation_tones: str | list[str],
                 stimulation_pulse_length: float,
                 ringdown_time: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = SpectroscopyModel,
                 stimulation_waveform_idx: int = 1,
                 stimulation_acquisition_idx: int = 2):
        
        self.stimulation_tones = make_it_list(stimulation_tones)
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='ACStarkSpectroscopy',
                         x_plot_label='Pulse Detuning',
                         x_plot_unit='MHz',
                         x_start=detuning_start,
                         x_stop=detuning_stop,
                         x_points=detuning_points,
                         y_plot_label='Stimulation Amplitude', 
                         y_plot_unit='arb', 
                         y_start=amp_start, 
                         y_stop=amp_stop, 
                         y_points=amp_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        self.stimulation_pulse_length = stimulation_pulse_length
        self.ringdown_time = ringdown_time
        self.stimulation_waveform_idx = stimulation_waveform_idx
        self.stimulation_acquisition_idx = stimulation_acquisition_idx

        self.stimulation_pulse_length_ns = round(stimulation_pulse_length / u.ns)
        self.ringdown_time_ns = round(ringdown_time / u.ns)
        
        assert set(self.stimulation_tones).issubset(set(self.tones)), 'ACSS: stimulation_tones do not exist.'
        assert self.resonator_pulse_length_ns + self.stimulation_pulse_length_ns <= 16384, \
            f'ACSS: The stimulation + readout pulse cannot exceed 16384ns.'


    def add_xinit(self):
        """
        We need to avoid the dependency injection here.
        """
        Spectroscopy.add_xinit(self)

        
    def add_yinit(self):
        """
        Here R6 is the amplitude of stimulation pulse.
        """
        super().add_yinit()
        
        for tone in self.stimulation_tones:
            y_start = self.gain_translator(self.y_start)
            self.sequences[tone]['program'] += f"""
                    move             {y_start},R6
            """


    def add_main(self):
        length = self.stimulation_pulse_length_ns + self.ringdown_time_ns
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
            
        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)
                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     R6,R6
                    reset_ph
                    play             {self.stimulation_waveform_idx},{self.stimulation_waveform_idx},{tof_ns}
                    acquire          {self.stimulation_acquisition_idx},R1,{length-tof_ns} 
                """

            elif tone in self.main_tones:
                step = self.frequency_translator(self.x_step)
                gain = round(self.cfg.variables[f'{tone}']['amp_180'] * 32768)
                gain_drag = round(gain * self.cfg.variables[f'{tone}']['DRAG_weight'])
                main = f"""
                #-----------Main-----------
                    wait             {self.stimulation_pulse_length_ns - self.qubit_pulse_length_ns}
                    set_freq         R4
                    set_awg_gain     {gain},{gain_drag}
                    play             0,1,{self.qubit_pulse_length_ns + self.ringdown_time_ns}
                    add              R4,{step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {length}
                """

            self.sequences[tone]['program'] += main


    def add_yvalue(self):
        y_step = self.gain_translator(self.y_step)
        for tone in self.stimulation_tones:  self.sequences[tone]['program'] += f"""
                    add              R6,{y_step},R6
        """


    def fit_data(self, x: list | np.ndarray = None, **fitting_kwargs):
        """
        We won't fit 2D data, instead, we treat each amp as an independent spectroscopy and fit it.
        See Scan.fit_data() as reference.
        """
        self.fit_result = {rr: [] for rr in self.readout_resonators}
        if self.fitmodel is None: return
        if x is None: x = self.x_values
        
        for i, rr in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']

            for j in range(self.y_points):
                try:
                    result = fit(input_data=self.measurement[rr]['to_fit'][level_index][j],
                                 x=x, fitmodel=self.fitmodel, t=self.qubit_pulse_length_ns * u.ns,
                                 **fitting_kwargs)
                    self.fit_result[rr].append(result)
                    
                    params = {v.name:{'value':v.value, 'stderr':v.stderr} for v in result.params.values()}
                    self.measurement[rr][f'fit_result_{j}'] = params
                    self.measurement[rr]['fit_model'] = str(result.model)
                except Exception:
                    self.fitting_traceback = traceback.format_exc()  # Return a string to debug.
                    print(f'Scan: Failed to fit {rr} {j}-th amp data. ')
                    self.measurement[rr][f'fit_result_{j}'] = None
                    self.measurement[rr]['fit_model'] = str(self.fitmodel)
    

    def plot_main(self, text_loc: str = 'lower right', dpi: int = 150):
        """
        Here we will save all the plot without showing in console or make them attributes.
        See Scan.plot_main() as reference.
        """
        for i, rr in enumerate(self.readout_resonators):
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']      

            if self.classification_enable:
                ylabel = fr'$P_{{\left|{self.level_to_fit[i]}\right\rangle}}$'
            else:
                ylabel = 'I-Q Coordinate (Rotated) [a.u.]'

            for j, amp in enumerate(self.y_values):
                fig, ax = plt.subplots(1, 1, dpi=dpi)
                ax.plot(self.x_values / self.x_unit_value, self.measurement[rr]['to_fit'][level_index][j], 'k.')
                ax.set(xlabel=self.x_plot_label + f'[{self.x_plot_unit}]', ylabel=ylabel, 
                       title=f'{self.datetime_stamp}, {self.scan_name}, {rr}, Amp{amp}')

                if self.measurement[rr][f'fit_result_{j}'] is not None: 
                    # Raise resolution of fit result for smooth plot.
                    x = np.linspace(self.x_start, self.x_stop, self.x_points * 3)  
                    y = self.fit_result[rr][j].eval(x=x)
                    ax.plot(x / self.x_unit_value, y, 'm-')
                    
                    fit_text = '\n'.join([f'{v.name} = {v.value:0.3g}' for v in self.fit_result[rr][j].params.values()])
                    ax.add_artist(AnchoredText(fit_text, loc=text_loc, prop={'color':'m'}))

                fig.savefig(os.path.join(self.data_path, f'{rr}_amp{j}.png'))
                plt.close('all')

    
    def plot_populations(self, dpi: int = 150):
        """
        Here we will save all the plot without showing in console or make them attributes.
        See Scan.plot_population() as reference.
        """
        for rr in self.readout_resonators:
            for j, amp in enumerate(self.y_values):
                fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=dpi)
                for i, level in enumerate(self.cfg[f'variables.{rr}/readout_levels']):
                    ax[0].plot(self.x_values / self.x_unit_value, self.measurement[rr]['PopulationNormalized_readout'][i][j], 
                               c=self.color_list[level], ls='-', marker='.', label=fr'$P_{{{level}}}$')
                    ax[1].plot(self.x_values / self.x_unit_value, self.measurement[rr]['PopulationCorrected_readout'][i][j], 
                               c=self.color_list[level], ls='-', marker='.', label=fr'$P_{{{level}}}$')

                xlabel = f'{self.x_plot_label}[{self.x_plot_unit}]'
                ax[0].set(xlabel=xlabel, ylabel='Uncorrected populations', ylim=(-0.05, 1.05))
                ax[1].set(xlabel=xlabel, ylabel='Corrected populations', ylim=(-0.05, 1.05))
                ax[0].legend()
                ax[1].legend()
                ax[0].set_title(f'{self.datetime_stamp}, {self.scan_name}, {rr}, Amp{amp}')
                fig.savefig(os.path.join(self.data_path, f'{rr}_Population_Amp{j}.png'))
                fig.clear()
                plt.close('all')


class IonizationSquareStimulation(Ionization):
    def set_waveforms_acquisitions(self):
        """
        Add the simulation waveform to sequence_dict.

        Note from Zihao(2024/07/26):
        When stimulation tone is one of the readout tones, the update method of sequence_dict is required.
        When stimulation tone start with "R" but not actually readout tones, it still works properly.
        The only drawback is we might have useless bins under acquisitions['readout'].
        """
        Scan.set_waveforms_acquisitions(self, add_special_waveforms=False)

        for tone in self.stimulation_tones:
            acquisitions = {'stimulation': {'num_bins': self.num_bins, 'index': self.stimulation_acquisition_idx}}
            self.sequences[tone]['acquisitions'].update(acquisitions)
    
            
    def add_main(self):
        tof_ns = round(self.cfg.variables['common/tof'] * 1e9)
        step = self.gain_translator(self.x_step)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)

                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_offs     R4,R4
                    reset_ph
                    upd_param        {tof_ns} 
                    acquire          {self.stimulation_acquisition_idx},R1,{self.stimulation_pulse_length_ns-tof_ns}
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                    add              R4,{step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {self.stimulation_pulse_length_ns + self.ringdown_time_ns}
                """

            self.sequences[tone]['program'] += main


class IonizationLengthScan(Scan2D, Ionization):
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 amp_start: float,
                 amp_stop: float,
                 amp_points: int,
                 length_start: float, 
                 length_stop: float, 
                 length_points: int,
                 stimulation_tones: str | list[str],
                 ringdown_time: float, 
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = None):
        
        self.stimulation_tones = make_it_list(stimulation_tones)
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_tones=readout_tones,
                         scan_name='IonizationLengthScan',
                         x_plot_label='Stimulation Amplitude',
                         x_plot_unit='arb',
                         x_start=amp_start,
                         x_stop=amp_stop,
                         x_points=amp_points,
                         y_plot_label='Stimulation Length', 
                         y_plot_unit='ns', 
                         y_start=length_start, 
                         y_stop=length_stop, 
                         y_points=length_points, 
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        self.ringdown_time = ringdown_time
        self.ringdown_time_ns = round(ringdown_time / u.ns)
        
        assert set(self.stimulation_tones).issubset(set(self.tones)), 'ILS: stimulation_tones do not exist.'
        assert 8 * u.ns < self.y_start < self.y_stop < 65536 * u.ns, \
            'ILS: All stimulation length must be in range (8, 65536) ns.'


    def set_waveforms_acquisitions(self):
        """
        Here we won't use any waveform or do any acquisiton for the stimulation tones.
        """
        super(Ionization, self).set_waveforms_acquisitions(add_special_waveforms=False)


    def add_yinit(self):
        """
        Here R6 is the length of stimulation pulse.
        """
        super().add_yinit()
        
        for tone in self.tones:
            self.sequences[tone]['program'] += f"""
                    move             {round(self.y_start / u.ns)},R6
            """


    def add_main(self):
        x_step = self.gain_translator(self.x_step)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)

                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_offs     R4,R4
                    reset_ph
                    sub              R6,8,R11
                    upd_param        8
                    wait             R11
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                    add              R4,{x_step},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             R6
                    wait             {self.ringdown_time_ns}
                    add              R4,{x_step},R4
                """

            self.sequences[tone]['program'] += main


    def add_yvalue(self):
        for tone in self.tones:  
            self.sequences[tone]['program'] += f"""
                    add              R6,{round(self.y_step / u.ns)},R6
            """

