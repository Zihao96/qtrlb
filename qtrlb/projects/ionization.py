import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from abc import ABCMeta, abstractmethod
from scipy.integrate import solve_ivp
from matplotlib.offsetbox import AnchoredText

import qtrlb.utils.units as u
from qtrlb.config.config import MetaManager
from qtrlb.calibration.calibration import Scan, Scan2D
from qtrlb.calibration.scan_classes import Spectroscopy
from qtrlb.utils.waveforms import get_waveform
from qtrlb.utils.general_utils import make_it_list
from qtrlb.processing.fitting import fit, SpectroscopyModel
PI = np.pi



def Kerr_oscillator(t, A, detuning, Kerr, kappa, Omega_0, slope=0, phase=0):
    """
    Right-hand side of the equation of motion of a driven damping Kerr resonator in the drive frame.
    Here we allow the drive amplitude Omega to ramp up linearly with a constant slope.
    """
    derivative = (
        -1j * 2*PI * detuning * A 
        -1j * 2*PI * Kerr * np.abs(A)**2 * A
        -PI * kappa * A 
        -1j * PI * (Omega_0 + slope * t) * np.exp(-1j * phase)
    )
    return derivative


def mean_field_amplitude_v4(Kerr: float, kappa: float, ramp_time: float, t_: float,
                            Omega_i: float, Omega_f: float, t_step: float,
                            ramp_ratio: float = None, detuning: float = None, Omega_ratio: float = 1):
    """
    ALL PARMETERS ARE IN CYCLIC FREQEUENCY.
    A shaped pulse with three stages: rampup, LZ, rampdown.
    All stages use same detuning and save phase, while each stages has its own drive amplitude.

    The rampup quickly prepare the resonator in to a targeting initial photon number n_i, \
        corresponding to steady state photons n_i and amplitude Omega_i.
    The LZ stage try to linearly increase photon number from n_i to n_f using a constant drive amplitude Omega_. \
        Typically Omega_ > Omega_f, otherwise photons will curve down rather than being linear.
    The rampdown quickly empty the resonator from a photon number n_f, \
        corresponding to steady state photons n_f and amplitude Omega_f.

    We leave the detuning and Omega_ratio as free parameters to be optimized.
    Omega_ratio is used to determine the Omega_ in the LZ stage, \
        and making sure it starts and ends at correct photon numbers
    detuningo is used to find a frequency and its rotating frame such that the Kerr effect \
        during three stages can be self-compensated and rampdown really empty the resonator.

    The detuning is defined as w_r - w_d here.
    To recover to v3 sequence, set Omega_i=Omega_f=Omega_.
    """
    if ramp_ratio is None: ramp_ratio = 1 / (1 - np.exp(-PI * kappa * ramp_time))
    if detuning is None: detuning = - Kerr * np.abs(Omega_i / kappa) ** 2
    Omega_up = Omega_i * ramp_ratio
    Omega_ = Omega_i * Omega_ratio
    Omega_down = Omega_f * (1 - ramp_ratio)

    # Rampup
    fun = lambda t, A: Kerr_oscillator(t, A, detuning, Kerr, kappa, Omega_up)
    t_up = np.linspace(0, ramp_time, round(ramp_time / t_step + 1))
    sol = solve_ivp(fun, t_span=[t_up[0], t_up[-1]], y0=[0j], method='RK45', t_eval=t_up)
    A_up = sol.y[0]

    # Large but no ramping amplitude, but linear ramping photons
    fun = lambda t, A: Kerr_oscillator(t, A, detuning, Kerr, kappa, Omega_)
    t_steady = np.linspace(0, t_, round(t_ / t_step + 1))
    sol = solve_ivp(fun, t_span=[t_steady[0], t_steady[-1]], y0=[sol.y[0][-1]], method='RK45', t_eval=t_steady)
    A_steady = sol.y[0]

    # Rampdown
    fun = lambda t, A: Kerr_oscillator(t, A, detuning, Kerr, kappa, Omega_down)
    t_down = np.linspace(0, ramp_time, round(ramp_time / t_step + 1))
    sol = solve_ivp(fun, t_span=[t_down[0], t_down[-1]], y0=[sol.y[0][-1]], method='RK45', t_eval=t_down)
    A_down = sol.y[0]

    return A_up, A_steady, A_down




class IonizationBase(Scan, metaclass=ABCMeta):
    """ Base class for Ionization type of experiments.
        User must overload the __init__, add_xinit and add_main for child class.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
                 stimulation_waveform: list = None,
                 stimulation_waveform_idx: int = 2,
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
                         stimulation_waveform=stimulation_waveform,
                         stimulation_waveform_idx=stimulation_waveform_idx,
                         stimulation_acquisition_idx=stimulation_acquisition_idx,
                         stimulation_pulse_length_ns=round(stimulation_pulse_length / u.ns),
                         ringdown_time_ns=round(ringdown_time / u.ns))


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
            if self.stimulation_waveform is None:
                self.stimulation_waveform = get_waveform(
                    length=self.stimulation_pulse_length_ns, 
                    shape=self.cfg[f'variables.{tone}/pulse_shape']
                )

            waveforms = {'stimulation': {'data': self.stimulation_waveform, 'index': self.stimulation_waveform_idx}}
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
        """
        Because upd_param does not accept registers, we set R11 = R4 - 8(ns).
        """
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


class IonizationLengthAmpScan(Scan2D, IonizationAmpSquarePulse):
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
    Sweep the spectroscopy frequency (x) and stimulation amplitude (y) in a typical ionization experiment:
    state preparation -> stimulation -> ringdown -> readout.
    The spectroscopy pulse is added just before the end of the simulation.
    This class was called as ACStarkSpectroscopy as a calibration for photon numbers.
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
        Spectroscopy.add_xinit(self)

        
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


class IonizationDelaySpectroscopy(Scan2D, IonizationAmpScan, Spectroscopy):
    """
    Sweep the spectroscopy frequency (x) and delay time (y) in a typical ionization experiment:
    state preparation -> stimulation -> ringdown -> readout.
    The spectroscopy pulse is added during stimulation and ringdown.
    This class should ALLOW arbitrary waveform for stimulation WITHOUT acquisition.
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
                 fitmodel: Model = SpectroscopyModel,
                 stimulation_waveform: list = None,
                 stimulation_waveform_idx: int = 2,
                 stimulation_acquisition_idx: int = 2):
        
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
                         stimulation_waveform=stimulation_waveform,
                         stimulation_waveform_idx=stimulation_waveform_idx,
                         stimulation_acquisition_idx=stimulation_acquisition_idx,
                         stimulation_pulse_length_ns=round(stimulation_pulse_length / u.ns),
                         ringdown_time_ns=round(ringdown_time / u.ns))


    def check_attribute(self):
        super().check_attribute()
        assert 0 <= self.y_start < self.y_stop < self.stimulation_pulse_length + self.ringdown_time, \
            "IDS: All delay time must be in range [0, stimulation_pulse_length + ringdown_time) ns."
    

    def add_xinit(self):
        Spectroscopy.add_xinit(self)

        
    def add_yinit(self):
        """
        Here R6 is the beginning time of the spectroscopy pulse.
        R6 = 0 means the spectroscopy pulse starts at the same time as the stimulation.
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
            length = self.stimulation_pulse_length_ns + self.ringdown_time_ns

            if tone in self.stimulation_tones:
                freq = round(self.cfg.variables[f'{tone}/mod_freq'] * 4)
                amp = round(self.cfg.variables[f'{tone}/amp'] * 32768)
                main = f"""
                #-----------Main-----------
                    set_freq         {freq}
                    set_awg_gain     {amp},{amp}
                    reset_ph
                    play             {self.stimulation_waveform_idx},{self.stimulation_waveform_idx},{length} 
                """

            elif tone in self.main_tones:
                gain = round(self.cfg.variables[f'{tone}']['amp_180'] * 32768)
                gain_drag = round(gain * self.cfg.variables[f'{tone}']['DRAG_weight'])
                step = self.frequency_translator(self.x_step)
                main = f"""
                #-----------Main-----------
                    move             {length - self.qubit_pulse_length_ns},R11
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
                    wait             {length}
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
        Plot the 2D results and spectroscopy fitting plot at each time points.
        Here we will save spectroscopy fitting plot without showing in console or make them attributes.
        See Scan.plot_main() and Scan2D.plot_main() as reference.
        """
        self.figures = {}

        for i, rr in enumerate(self.readout_resonators):
            data = self.measurement[rr]['to_fit']
            n_subplots = len(data)
            xlabel = self.x_plot_label + f'[{self.x_plot_unit}]'
            ylabel = self.y_plot_label + f'[{self.y_plot_unit}]'
            title = f'{self.datetime_stamp}, {self.scan_name}, {rr}'

            # 2D plot
            fig, ax = plt.subplots(1, n_subplots, figsize=(7 * n_subplots, 8), dpi=dpi)
            for l in range(n_subplots):
                level = self.cfg[f'variables.{rr}/readout_levels'][l]
                this_title = title + fr', $P_{{{level}}}$' if self.classification_enable else title
                extent = [
                    np.min(self.x_values) / self.x_unit_value, np.max(self.x_values) / self.x_unit_value, 
                    np.min(self.y_values) / self.y_unit_value, np.max(self.y_values) / self.y_unit_value
                ]
                image = ax[l].imshow(data[l], cmap='Blues', interpolation='none', aspect='auto', 
                                     vmin=0, vmax=1, origin='lower', extent=extent)
                ax[l].set(title=this_title, xlabel=xlabel, ylabel=ylabel)
                fig.colorbar(image, ax=ax[l], label='Probability/Coordinate', location='top')
                
            fig.savefig(os.path.join(self.data_path, f'{rr}.png'))
            self.figures[rr] = fig

            # Spectroscopy fitting plot.
            level_index = self.level_to_fit[i] - self.cfg[f'variables.{rr}/lowest_readout_levels']      
            if self.classification_enable:
                ylabel = fr'$P_{{\left|{self.level_to_fit[i]}\right\rangle}}$'
            else:
                ylabel = 'I-Q Coordinate (Rotated) [a.u.]'

            for j, time in enumerate(self.y_values):
                fig, ax = plt.subplots(1, 1, dpi=dpi)
                ax.plot(self.x_values / self.x_unit_value, self.measurement[rr]['to_fit'][level_index][j], 'k.')
                ax.set(xlabel=xlabel, ylabel=ylabel, 
                       title=f'{title}, Time{round(time / self.y_unit_value)}({self.y_plot_unit})')

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


class IonizationSteadyState(IonizationBase):
    """
    Sweep the length of steady state drive in a typical ionization experiment:
    state preparation -> stimulation (rampup-steady_state-reset_rampdown) -> free_ringdown -> readout.
    This class allows detuned step-wise stimulation (3 stages) and does not allow acquisition or arbitrary waveform.
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 length_start: float,
                 length_stop: float,
                 length_points: int,
                 stimulation_tones: str | list[str],
                 ramp_time: float,
                 ramp_ratio: float,
                 ringdown_time: float,
                 detuning_coeff: float = 0.0,
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
                         scan_name='IonizationSteadyState',
                         x_plot_label='Drive length',
                         x_plot_unit='us',
                         x_start=length_start,
                         x_stop=length_stop,
                         x_points=length_points,
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         stimulation_tones=make_it_list(stimulation_tones),
                         ramp_time=ramp_time,
                         ramp_time_ns=round(ramp_time / u.ns),
                         ramp_ratio=ramp_ratio,
                         ringdown_time=ringdown_time,
                         ringdown_time_ns=round(ringdown_time / u.ns),
                         detuning_coeff=detuning_coeff)


    def check_attribute(self):
        super().check_attribute()
        assert 8 * u.ns < self.x_start < self.x_stop < 65536 * u.ns, \
            'ISS: All drive length must be in range (8, 65536) ns.'


    def add_xinit(self):
        """
        Here R4 is the drive length.
        """
        super().add_xinit()
        for tone in self.tones:
            self.sequences[tone]['program'] += f"""
                    move             {round(self.x_start * 1e9)},R4
            """

    
    def add_main(self):
        step_ns = round(self.x_step * 1e9)

        for tone in self.tones:
            if tone in self.stimulation_tones:
                amp = self.cfg.variables[f'{tone}/amp']
                detuning = self.detuning_coeff * amp**2
                freq = round((self.cfg.variables[f'{tone}/mod_freq'] + detuning) * 4)

                # Calculate the three amplitudes.
                waveform = np.array((self.ramp_ratio, 1.0, -1 * self.ramp_ratio + 1)) / self.ramp_ratio
                up, hold, down = np.round(waveform * amp * 32768).astype(int)

                main = f"""
                #-----------Main-----------
                    sub              R4,8,R11
                    set_freq         {freq}

                    # Rampup
                    set_awg_offs     {up},{up}
                    reset_ph
                    upd_param        {self.ramp_time_ns} 

                    # Steady state
                    set_awg_offs     {hold},{hold}
                    upd_param        8
                    wait             R11

                    # Reset rampdown
                    set_awg_offs     {down},{down}
                    upd_param        {self.ramp_time_ns}

                    # Free ringdown
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                    add              R4,{step_ns},R4
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {self.ramp_time_ns}
                    wait             R4
                    wait             {self.ramp_time_ns}
                    wait             {self.ringdown_time_ns}
                    add              R4,{step_ns},R4
                """

            self.sequences[tone]['program'] += main


class IonizationSteadyStateSpectroscopy(IonizationDelaySpectroscopy):
    """
    Sweep the spectroscopy frequency (x) and delay time (y) in a steady-state ionization experiment:
    state preparation -> stimulation (rampup-steady_state-reset_rampdown) -> free_ringdown -> readout.
    The spectroscopy pulse is added during stimulation and free_ringdown.
    This class allows detuned step-wise stimulation (3 stages) and does not allow acquisition or arbitrary waveform.
    The __init__ and check_attribute will use parents started from Scan2D.
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
                 steady_state_length: float,
                 ramp_time: float,
                 ramp_ratio: float,
                 ringdown_time: float,
                 detuning_coeff: float = 0.0,
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = SpectroscopyModel):
        
        super(IonizationDelaySpectroscopy, self).__init__(
            cfg=cfg, 
            drive_qubits=drive_qubits,
            readout_tones=readout_tones,
            scan_name='IonizationSteadyStateSpectroscopy',
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
            steady_state_length=steady_state_length,
            steady_state_length_ns=round(steady_state_length / u.ns),
            ramp_time=ramp_time,
            ramp_time_ns=round(ramp_time / u.ns),
            ramp_ratio=ramp_ratio,
            ringdown_time=ringdown_time,
            ringdown_time_ns=round(ringdown_time / u.ns),
            detuning_coeff=detuning_coeff
        )
        

    def check_attribute(self):
        super(IonizationDelaySpectroscopy, self).check_attribute()
        assert 0 <= self.y_start < self.y_stop < self.steady_state_length + 2 * self.ramp_time + self.ringdown_time, \
            "ISSS: All delay time must be in range [0, steady_state_length + 2*ramp_time + ringdown_time) ns."


    def set_waveforms_acquisitions(self):
        super(IonizationAmpScan, self).set_waveforms_acquisitions()


    def add_main(self):
        """
        Here R11 is the total wait time excluding the length of the spectroscopy pulse.
        R12 is the wait time after the end of the spectroscopy pulse. R12 = R11 - R6.
        So the sequence is R6--Spectroscopy--R12.
        """        
        for tone in self.tones:
            length = self.steady_state_length_ns + 2 * self.ramp_time_ns + self.ringdown_time_ns

            if tone in self.stimulation_tones:
                amp = self.cfg.variables[f'{tone}/amp']
                detuning = self.detuning_coeff * amp**2
                freq = round((self.cfg.variables[f'{tone}/mod_freq'] + detuning) * 4)

                # Calculate the three amplitude.
                waveform = np.array((self.ramp_ratio, 1.0, -1 * self.ramp_ratio + 1)) / self.ramp_ratio
                up, hold, down = np.round(waveform * amp * 32768).astype(int)

                main = f"""
                #-----------Main-----------
                    set_freq         {freq}

                    # Rampup
                    set_awg_offs     {up},{up}
                    reset_ph
                    upd_param        {self.ramp_time_ns} 

                    # Steady state
                    set_awg_offs     {hold},{hold}
                    upd_param        {self.steady_state_length_ns}

                    # Reset rampdown
                    set_awg_offs     {down},{down}
                    upd_param        {self.ramp_time_ns}

                    # Free ringdown
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                """

            elif tone in self.main_tones:
                gain = round(self.cfg.variables[f'{tone}']['amp_180'] * 32768)
                gain_drag = round(gain * self.cfg.variables[f'{tone}']['DRAG_weight'])
                step = self.frequency_translator(self.x_step)
                main = f"""
                #-----------Main-----------
                    move             {length - self.qubit_pulse_length_ns},R11
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
                    wait             {length}
                """

            self.sequences[tone]['program'] += main


class IonizationLandauZener(IonizationBase):
    """
    The Landau-Zener experiments of transmon ionization.
    It's not really a parameter sweep, but a single point experiment.
    We must sweep adiabticity ourside the qtrlb.Scan framework.
    Sequence: \
    state preparation -> stimulation (rampup-steady_state-LandauZener-reset_rampdown) -> free_ringdown -> readout.
    This class allows detuned step-wise stimulation (4 stages) and does not allow acquisition or arbitrary waveform.

    THE STEADY-STATE STAGE HAS NOT BEEN IMPLEMENTED YET (because in this detuning n_i is not stable).
    We can set it to 0 now.
    """
    def __init__(self,
                 cfg: MetaManager, 
                 drive_qubits: str | list[str],
                 readout_tones: str | list[str],
                 stimulation_tones: str | list[str],
                 ramp_time: float,
                 ramp_ratio: float,
                 steady_state_length: float,
                 LandauZener_length: float,
                 ringdown_time: float,
                 amp_i: float,
                 amp_f: float,
                 amp_LZratio: float,
                 detuning: float,
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
                         scan_name='IonizationLandauZener',
                         x_plot_label='',
                         x_plot_unit='arb',
                         x_start=1,
                         x_stop=1,
                         x_points=1,
                         subspace=subspace,
                         main_tones=main_tones,
                         pre_gate=pre_gate,
                         post_gate=post_gate,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel,
                         stimulation_tones=make_it_list(stimulation_tones),
                         ramp_time=ramp_time,
                         ramp_time_ns=round(ramp_time / u.ns),
                         ramp_ratio=ramp_ratio,
                         steady_state_length=steady_state_length,
                         steady_state_length_ns=round(steady_state_length / u.ns),
                         LandauZener_length=LandauZener_length,
                         LandauZener_length_ns=round(LandauZener_length / u.ns),
                         ringdown_time=ringdown_time,
                         ringdown_time_ns=round(ringdown_time / u.ns),
                         amp_i=amp_i,
                         amp_f=amp_f,
                         amp_LZratio=amp_LZratio,
                         detuning=detuning)


    def check_attribute(self):
        super().check_attribute()
        assert 8 * u.ns < self.LandauZener_length, 'ILZ: The Landau-Zener length must be longer than 8 ns.'


    def add_xinit(self):
        return Scan.add_xinit(self)


    def add_main(self):
        for tone in self.tones:
            if tone in self.stimulation_tones:
                # 3-stages steady state drive amplitudes.
                waveform = np.array((self.ramp_ratio, 1.0, -1 * self.ramp_ratio + 1)) / self.ramp_ratio
                up_i, hold_i, down_i = np.round(waveform * self.amp_i * 32768).astype(int)
                up_f, hold_f, down_f = np.round(waveform * self.amp_f * 32768).astype(int)
                LZ = np.round(self.amp_i / self.ramp_ratio * self.amp_LZratio * 32768).astype(int)
                freq = round((self.cfg.variables[f'{tone}/mod_freq'] + self.detuning) * 4)

                main = f"""
                    set_freq         {freq}

                    # Rampup
                    set_awg_offs     {up_i},{up_i}
                    reset_ph
                    upd_param        {self.ramp_time_ns} 

                    # # Steady-state   
                    # set_awg_offs     {hold_i},{hold_i}
                    # upd_param        {self.steady_state_length_ns}

                    # Landau-Zener
                    set_awg_offs     {LZ},{LZ}
                    upd_param        {self.LandauZener_length_ns}

                    # Reset rampdown
                    set_awg_offs     {down_f},{down_f}
                    upd_param        {self.ramp_time_ns}

                    # Free ringdown
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                """

            else:
                main = f"""
                #-----------Main-----------
                    wait             {self.ramp_time_ns}
                    wait             {self.LandauZener_length_ns}
                    wait             {self.ramp_time_ns}
                    wait             {self.ringdown_time_ns}
                """

            self.sequences[tone]['program'] += main


class IonizationLandauZenerSpectroscopy(IonizationDelaySpectroscopy):
    """
    Sweep the spectroscopy frequency (x) and delay time (y) in a Landau-Zener ionization experiment:
    state preparation -> stimulation (rampup-steady_state-LandauZener-reset_rampdown) -> free_ringdown -> readout.
    This class allows detuned step-wise stimulation (4 stages) and does not allow acquisition or arbitrary waveform.
    The spectroscopy pulse is added during stimulation and free_ringdown.
    The __init__ and check_attribute will use parents started from Scan2D.

    THE STEADY-STATE STAGE HAS NOT BEEN IMPLEMENTED YET (because in this detuning n_i is not stable).
    We can set it to 0 now.
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
                 ramp_time: float,
                 ramp_ratio: float,
                 steady_state_length: float,
                 LandauZener_length: float,
                 ringdown_time: float,
                 amp_i: float,
                 amp_f: float,
                 amp_LZratio: float,
                 detuning: float,
                 subspace: str | list[str] = None,
                 main_tones: str | list[str] = None,
                 pre_gate: dict[str: list[str]] = None,
                 post_gate: dict[str: list[str]] = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list[int] = None,
                 fitmodel: Model = SpectroscopyModel):
        
        super(IonizationDelaySpectroscopy, self).__init__(
            cfg=cfg, 
            drive_qubits=drive_qubits,
            readout_tones=readout_tones,
            scan_name='IonizationLandauZenerSpectroscopy',
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
            ramp_time=ramp_time,
            ramp_time_ns=round(ramp_time / u.ns),
            ramp_ratio=ramp_ratio,
            steady_state_length=steady_state_length,
            steady_state_length_ns=round(steady_state_length / u.ns),
            LandauZener_length=LandauZener_length,
            LandauZener_length_ns=round(LandauZener_length / u.ns),
            ringdown_time=ringdown_time,
            ringdown_time_ns=round(ringdown_time / u.ns),
            amp_i=amp_i,
            amp_f=amp_f,
            amp_LZratio=amp_LZratio,
            detuning=detuning)
        

    def check_attribute(self):
        super(IonizationDelaySpectroscopy, self).check_attribute()
        assert (0 <= self.y_start < self.y_stop 
                < 2 * self.ramp_time + self.steady_state_length + self.LandauZener_length + self.ringdown_time), \
            "ILZS: All delay time must be in range \
            [0, 2*ramp_time + steady_state_length + LandauZener_length + ringdown_time) ns."


    def set_waveforms_acquisitions(self):
        super(IonizationAmpScan, self).set_waveforms_acquisitions()


    def add_main(self):
        """
        Here R11 is the total wait time excluding the length of the spectroscopy pulse.
        R12 is the wait time after the end of the spectroscopy pulse. R12 = R11 - R6.
        So the sequence is R6--Spectroscopy--R12.
        """        
        for tone in self.tones:
            length = (2 * self.ramp_time_ns
                      + self.steady_state_length_ns
                      + self.LandauZener_length_ns
                      + self.ringdown_time_ns)

            if tone in self.stimulation_tones:
                # 3-stages steady state drive amplitudes.
                waveform = np.array((self.ramp_ratio, 1.0, -1 * self.ramp_ratio + 1)) / self.ramp_ratio
                up_i, hold_i, down_i = np.round(waveform * self.amp_i * 32768).astype(int)
                up_f, hold_f, down_f = np.round(waveform * self.amp_f * 32768).astype(int)
                LZ = np.round(self.amp_i / self.ramp_ratio * self.amp_LZratio * 32768).astype(int)
                freq = round((self.cfg.variables[f'{tone}/mod_freq'] + self.detuning) * 4)

                main = f"""
                    set_freq         {freq}

                    # Rampup
                    set_awg_offs     {up_i},{up_i}
                    reset_ph
                    upd_param        {self.ramp_time_ns} 

                    # # Steady-state   
                    # set_awg_offs     {hold_i},{hold_i}
                    # upd_param        {self.steady_state_length_ns}

                    # Landau-Zener
                    set_awg_offs     {LZ},{LZ}
                    upd_param        {self.LandauZener_length_ns}

                    # Reset rampdown
                    set_awg_offs     {down_f},{down_f}
                    upd_param        {self.ramp_time_ns}

                    # Free ringdown
                    set_awg_offs     0,0
                    upd_param        {self.ringdown_time_ns}
                """

            elif tone in self.main_tones:
                gain = round(self.cfg.variables[f'{tone}']['amp_180'] * 32768)
                gain_drag = round(gain * self.cfg.variables[f'{tone}']['DRAG_weight'])
                step = self.frequency_translator(self.x_step)
                main = f"""
                #-----------Main-----------
                    move             {length - self.qubit_pulse_length_ns},R11
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
                    wait             {length}
                """

            self.sequences[tone]['program'] += main