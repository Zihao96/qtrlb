"Living as realist, dreaming as idealist, always be curious, and never give up. --Z"

import os

from qtrlb.calibration.calibration import Scan, Scan2D
from qtrlb.calibration.scan_classes import DriveAmplitudeScan, Spectroscopy, RabiScan, DebugRabi, T1Scan, \
    RamseyScan, EchoScan, LevelScan, CalibrateClassification, JustGate, CalibrateTOF, CheckBlobShift, \
    QNDnessCheck, TwoToneROCalibration, MultitoneROCalibration
from qtrlb.calibration.scan2d_classes import ChevronScan, AmplitudeDetuningScan, ReadoutFrequencyScan, \
    ReadoutAmplitudeScan, ReadoutLengthAmpScan, DRAGWeightScan
from qtrlb.calibration.mixer_correction import MixerCorrection, MixerAutoCorrection
from qtrlb.calibration.autotune import autotune

from qtrlb.benchmark.randomized_benchmarking import RB1QB, RB1QBDetuningSweep, RB1QBAmp180Sweep, \
    RB1QBAmp90Sweep, RB1QBDRAGWeightSweep
from qtrlb.benchmark.state_tomography import StateTomography, SingleQuditStateTomography

from qtrlb.projects.ionization import IonizationAmpScan, IonizationAmpSquarePulse, IonizationRingDownScan, \
    IonizationLengthAmpScan, IonizationAmpSpectroscopy, IonizationDelaySpectroscopy, Ionization, \
    IonizationRingDown, ACStarkSpectroscopy, IonizationSquareStimulation, IonizationLengthScan

from qtrlb.processing.fitting import QuadModel, SinModel, ChevronModel, ResonatorHangerTransmissionModel

from qtrlb.config.variable_manager import VariableManager
from qtrlb.config.DAC_manager import DACManager
from qtrlb.config.process_manager import ProcessManager
from qtrlb.config.data_manager import DataManager
from qtrlb.config.gate_manager import GateManager
from qtrlb.config.config import Config, MetaManager

from qtrlb.utils.N9010A import N9010A




def begin_measurement_session(working_dir: str, 
                              variable_suffix: str = '',
                              test_mode: bool = False):
    """
    Instantiate all managers along with MetaManager, then load them.
    Return the instance of MetaManager.
    Please do not place this function and all import above at same file as MetaManager (circular import).
    """
    yamls_path = os.path.join(working_dir, 'Yamls')
    
    varman = VariableManager(yamls_path, variable_suffix)
    dacman = DACManager(yamls_path, varman, test_mode)
    processman = ProcessManager(yamls_path, varman)
    dataman = DataManager(yamls_path, varman)
    gateman = GateManager(yamls_path, varman)
    
    cfg = MetaManager(manager_dict={
                        'variables':varman,
                        'DAC':dacman,
                        'process':processman,
                        'data':dataman,
                        'gates':gateman
                        },
                      working_dir=working_dir)
    return cfg