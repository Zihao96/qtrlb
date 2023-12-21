import qtrlb.utils.units as u
import matplotlib.pyplot as plt
from qtrlb.config.config import MetaManager
from qtrlb.calibration.calibration import Scan
from qtrlb.calibration.scan_classes import RamseyScan, DriveAmplitudeScan
from qtrlb.calibration.scan2d_classes import DRAGWeightScan
from qtrlb.processing.fitting import QuadModel
from qtrlb.utils.misc import tone_to_qudit




def autotune(
        cfg: MetaManager, 
        drive_qubits: str | list[str], 
        readout_tones: str | list[str], 
        subspace: str | list[str], 
        level_to_fit: int | list[int], 
        rams_length: float, 
        rams_AD: float,
        normalize_subspace: bool = False,
        show_plot: bool = True,
        verbose: bool = False,
        **autotune_kwargs) -> tuple[Scan]:
    """
    A fine autotune for one subspace of a qudit by running two Ramsey with a DAS and a DWS.
    It only helps us calibrate small fluctuation on frequency and amplitude from day to day.
    Change hardware configuration and cooldown/warmup fridge won't be considered here.
    autotune_kwargs take arguments common to all four scans here, such as pre_gate/main_tones
    """
    if 'main_tones' in autotune_kwargs:
        main_tones = Scan.make_it_list(autotune_kwargs['main_tones'])
        main_tone = main_tones[0]
    else: 
        main_tone = f'{drive_qubits}/{subspace}'

    rr = tone_to_qudit(Scan.make_it_list(readout_tones)[0])
    amp_180 = cfg[f'variables.{main_tone}/amp_180']
    weight = cfg[f'variables.{main_tone}/DRAG_weight']

    # Instantiate the classes.
    ramsp = RamseyScan(cfg, drive_qubits, readout_tones, subspace=subspace, 
                       length_start=0, length_stop=rams_length, length_points=41, 
                       artificial_detuning=+rams_AD, level_to_fit=level_to_fit, **autotune_kwargs)

    ramsn = RamseyScan(cfg, drive_qubits, readout_tones, subspace=subspace, 
                       length_start=0, length_stop=rams_length, length_points=41, 
                       artificial_detuning=-rams_AD, level_to_fit=level_to_fit, **autotune_kwargs)

    das = DriveAmplitudeScan(cfg, drive_qubits, readout_tones, subspace=subspace, 
                             amp_start=amp_180*0.85/0.9, amp_stop=amp_180*0.95/0.9, amp_points=41, 
                             error_amplification_factor=9, fitmodel=QuadModel, 
                             level_to_fit=level_to_fit, **autotune_kwargs)
    
    dws = DRAGWeightScan(cfg, drive_qubits, readout_tones, subspace=subspace, 
                         weight_start=weight-0.3, weight_stop=weight+0.3, weight_points=41, 
                         level_to_fit=level_to_fit, **autotune_kwargs)
    
    ramsp.run(f'AD+{round(rams_AD/u.kHz)}kHz_autotune')
    ramsn.run(f'AD-{round(rams_AD/u.kHz)}kHz_autotune')
    if normalize_subspace:
        plt.close(ramsp.figures[rr])
        plt.close(ramsn.figures[rr])
        ramsp.normalize_subspace_population()
        ramsn.normalize_subspace_population()
    freq_p = ramsp.fit_result[rr].params['freq'].value
    freq_n = ramsn.fit_result[rr].params['freq'].value
    cfg[f'variables.{main_tone}/freq'] -= round((freq_p - freq_n) / 2)
    cfg.save(verbose=verbose)
    cfg.load()  # Load will help to generate correct mod_freq.

    das.run('EAx9_autotune')
    cfg[f'variables.{main_tone}/amp_180'] = das.fit_result[rr].params['x0'].value
    cfg[f'variables.{main_tone}/amp_90'] = das.fit_result[rr].params['x0'].value / 2 
    cfg.save(verbose=verbose)
    cfg.load()

    dws.run('autotune')
    cfg[f'variables.{main_tone}/DRAG_weight'] = dws.fit_result[rr].params['x0'].value
    cfg.save(verbose=verbose)
    cfg.load()

    if not show_plot: plt.close('all')
    return ramsp, ramsn, das, dws

