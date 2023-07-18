#%% Introduction

"""
Welcome to Qtrlb.
This example shows templates about how to run different built-in scans.
"""

#%% Initialize the cfg(MetaManager), no need for setting console working directory.

import numpy as np
import qtrlb as qtb
import qtrlb.utils.units as u
PI = np.pi

working_dir = r'C:\Users\machie\Box\Blok-Lab-Server\Projects\qtrlb_working_dir'
cfg = qtb.begin_measurement_session(working_dir=working_dir, variable_suffix='QUDIT', test_mode=False)
qblox = cfg.DAC.qblox


#%% Define useful variables

drive_qubits = 'Q2'
readout_resonators = 'R2'
subspace = '01'
level_to_fit = 1


#%% Rabi
rs = qtb.RabiScan(cfg, drive_qubits, readout_resonators, subspace=subspace, level_to_fit=level_to_fit)
rs.run('example')


#%% T1
t1s = qtb.T1Scan(cfg, drive_qubits, readout_resonators, 
                 length_start=0, length_stop=200*u.us, length_points=41, level_to_fit=level_to_fit)
t1s.run('example')


#%% Echo
echo = qtb.EchoScan(cfg, drive_qubits, readout_resonators, 
                    length_start=0, length_stop=200*u.us, length_points=41)
echo.run('example')


#%% Ramsey
# Be careful with the length. 4's multiple.
ramsp = qtb.RamseyScan(cfg, drive_qubits, readout_resonators, subspace=subspace, 
                       length_start=0, length_stop=6*u.us, length_points=41, 
                       artificial_detuning=+0.5*u.MHz, level_to_fit=level_to_fit)
ramsp.run('AD+500kHz_example')

ramsn = qtb.RamseyScan(cfg, drive_qubits, readout_resonators, subspace=subspace, 
                       length_start=0, length_stop=6*u.us, length_points=41, 
                       artificial_detuning=-0.5*u.MHz, level_to_fit=level_to_fit)
ramsn.run('AD-500kHz_example')


#%% DriveAmplitudeScan
das = qtb.DriveAmplitudeScan(cfg, drive_qubits, readout_resonators, subspace=subspace, 
                             amp_start=0.114, amp_stop=0.132, amp_points=41, error_amplification_factor=9, 
                             fitmodel=qtb.QuadModel, level_to_fit=level_to_fit)
das.run('EAx9_example')


#%% Classification
cal = qtb.CalibrateClassification(cfg, drive_qubits, readout_resonators, 
                                  level_start=0, level_stop=3, n_seqloops=5000, save_cfg=False)
cal.run('example')

print([cal.measurement[r]['ReadoutFidelity'] for r in cal.readout_resonators])


#%% ChevronScan
chev = qtb.ChevronScan(cfg, drive_qubits, readout_resonators, subspace=subspace, 
                       detuning_start=-40e6, detuning_stop=40e6, detuning_points=81, 
                       length_start=0, length_stop=320e-9, length_points=81, n_seqloops=10)
chev.run('example', n_pyloops=50)


#%% ReadoutFrequencyScan
rfs = qtb.ReadoutFrequencyScan(cfg, drive_qubits, readout_resonators, 
                               level_start=0, level_stop=5, 
                               detuning_start=-1.5*u.MHz, detuning_stop=+1.5*u.MHz, detuning_points=151, 
                               n_seqloops=100)
rfs.tones.extend(['Q4/01', 'Q4/12'])
rfs.run('example', n_pyloops=10)


#%% ReadoutAmplitudeScan
ras = qtb.ReadoutAmplitudeScan(cfg, drive_qubits, readout_resonators, 
                               level_start=0, level_stop=3, 
                               amp_start=0, amp_stop=0.2, amp_points=41, n_seqloops=500)
ras.run('example', n_pyloops=2)


#%% RLAS

rlas = qtb.ReadoutLengthAmpScan(cfg, drive_qubits, readout_resonators, 
                                level_start=0, level_stop=3, 
                                amp_start=0, amp_stop=0.3, amp_points=61, 
                                length_stop=6000*u.ns, length_points=60, n_seqloops=500)
rlas.run('example', n_pyloops=2)


#%% JustGate
fun = qtb.JustGate(cfg, drive_qubits, readout_resonators, just_gate={}, lengths=None, n_seqloops=3000)
fun.run('example')

    
#%% RB1QB
rb = qtb.RB1QB(cfg, drive_qubits, readout_resonators, 
               n_gates_start=0, n_gates_stop=900, n_gates_points=11, n_random=30)
rb.run('example')


#%% DRAGWeightScan

dws = qtb.DRAGWeightScan(cfg, drive_qubits, readout_resonators, subspace=subspace, 
                         weight_start=-0.3, weight_stop=0.3, weight_points=41, level_to_fit=level_to_fit)
dws.run('example')


#%% CalibrateTOF

# tof.raw_data will have shape (2, n_pyloops, 16384)
tof = qtb.CalibrateTOF(cfg, drive_qubits, readout_resonators)
tof.run('example', n_pyloops=1000)


#%% MixerCorrection

mxc = qtb.MixerCorrection(cfg, 'Q4', subspace='01', amp=0.4, waveform_length=40)
mxc.run()

# To stop and save parameter.
mxc.stop(save_cfg=True)

# To narrow down the range of DC offset.
mxc.stop(save_cfg=False)
mxc.create_ipywidget(offset0_min=-5, offset0_max=5, offset1_min=-6, offset1_max=6)


#%% Conditional Ramsey for measuring static ZZ coupling.

# We can switch control and target to see the difference.
# Also change gate to 'I' for comparison.
control = 'Q3'
target = 'Q2'
resonators = ['R2', 'R3']
pre_gate = {control: ['X180_01']}  

ramsp = qtb.RamseyScan(cfg, drive_qubits=[control, target], readout_resonators=resonators, 
                       length_start=0, length_stop=1.6*u.us, length_points=41, 
                       artificial_detuning=+3*u.MHz, main_tones=[f'{target}/01'], pre_gate=pre_gate)

ramsn = qtb.RamseyScan(cfg, drive_qubits=[control, target], readout_resonators=resonators, 
                       length_start=0, length_stop=1.6*u.us, length_points=41, 
                       artificial_detuning=-3*u.MHz, main_tones=[f'{target}/01'], pre_gate=pre_gate)

ramsp.run('AD+3MHz_PreXGate')
ramsn.run('AD-3MHz_PreXGate')


#%% Run autotune