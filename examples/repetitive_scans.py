#%% Introduction

"""
Welcome to Qtrlb.
This example shows how to construct some long-time repetitive experiments.
"""

#%% Initialize the cfg(MetaManager), no need for setting console working directory

import gc
import psutil
import qtrlb as qtb
import qtrlb.utils.units as u
import matplotlib.pyplot as plt
from datetime import datetime

# We expect a Yamls folder under this path and yamls files living inside it.
working_dir = r'C:\Users\machie\Box\Blok-Lab-Server\Projects\qtrlb_working_dir'

cfg = qtb.begin_measurement_session(working_dir=working_dir, variable_suffix='QUDIT', test_mode=False)
qblox = cfg.DAC.qblox


#%% Define useful variables

drive_qubits = 'Q2'
readout_resonators = 'R2'
subspace = '01'
level_to_fit = 1


#%% Autotune
# This example is from 20230716 for precisely calibrate all transition frequency of QUDIT Q2 of design round 2.
# I did a quick hack to make Q2/56 become Q4/01 because we reach the limit of +- 500MHz mod_freq.
# And R2 is for reading our |0>, |1>, |2>, |3>; R4 is for readout same transmon but |3>, |4>, |5>, |6>
# I did it without classification. So level_to_fit is not really 3/4, it's the number substracted by lowest level.

# Calibrate first four levels |0> to |3>
readout_resonators = 'R2'
qtb.autotune(cfg, drive_qubits, readout_resonators, subspace='01', level_to_fit=1, rams_length=40*u.us, rams_AD=100*u.kHz)
qtb.autotune(cfg, drive_qubits, readout_resonators, subspace='12', level_to_fit=0, rams_length=20*u.us, rams_AD=200*u.kHz)
qtb.autotune(cfg, drive_qubits, readout_resonators, subspace='23', level_to_fit=1, rams_length=8*u.us, rams_AD=500*u.kHz)


# Calibrate second four levels |3> to |6>
readout_resonators = 'R4'
qtb.autotune(cfg, drive_qubits, readout_resonators, subspace='34', level_to_fit=4, rams_length=8*u.us, rams_AD=500*u.kHz)
qtb.autotune(cfg, drive_qubits, readout_resonators, subspace='45', level_to_fit=3, rams_length=8*u.us, rams_AD=500*u.kHz)
qtb.autotune(cfg, drive_qubits=['Q2', 'Q4'], readout_resonators=readout_resonators, subspace=['45', '01'], level_to_fit=4, 
             rams_length=2*u.us, rams_AD=2*u.MHz, main_tones=['Q4/01'], pre_gate={'Q2': [f'X180_{i}{i+1}' for i in range(5)]})
# The |5>-|6> transition is tricky


#%% Overnight coherence Scan 
# This example is from 20230715 higher level coherence of QUDIT Q2 of design round 2.
# I did a quick hack to make Q2/56 become Q4/01 because we reach the limit of +- 500MHz mod_freq.
# And R2 is for reading our |0>, |1>, |2>, |3>; R4 is for readout same transmon but |3>, |4>, |5>, |6>
# To make the classification of second readout tone work normally, I write the CrazyCC below. 


# Before running the overnight coherence scan, we should calibrate the drive/readout pulse.
# We should also run the whole scans at least once to choose proper length_stop, artificial_detuning, etc.
total_repetition = 100


class CrazyCC(qtb.CalibrateClassification):
    def add_main(self):
        """
        Here we add all PI gate to our sequence program based on level_stop.
        We will use R4 to represent level and jlt instruction to skip later PI gate.
        """
        for tone in self.tones: self.sequences[tone]['program'] += """
                #-----------Main-----------
                    jlt              R4,1,@end_main    
        """         

        for level in range(self.x_stop):
            gate = {q: [f'X180_{level}{level+1}'] for q in self.drive_qubits} if level != 5 else {'Q4': [f'X180_01']}
            self.add_gate(gate, name=f'XPI{level}{level+1}')
            
            for tone in self.tones: self.sequences[tone]['program'] += f"""
                    jlt              R4,{level+2},@end_main    
        """
            
        for tone in self.tones: self.sequences[tone]['program'] += """
        end_main:   add              R4,1,R4    
        """


cal_0123 = qtb.CalibrateClassification(cfg, drive_qubits, readout_resonators='R2', 
                                       level_start=0, level_stop=3, 
                                       n_seqloops=5000, save_cfg=True)

t1s_01 = qtb.T1Scan(cfg, drive_qubits, readout_resonators='R2', subspace='01', 
                    length_start=0, length_stop=200*u.us, length_points=41, level_to_fit=1)

ramsp_01 = qtb.RamseyScan(cfg, drive_qubits, readout_resonators='R2', subspace='01', 
                          length_start=0, length_stop=40*u.us, length_points=41, 
                          artificial_detuning=0.1*u.MHz, level_to_fit=1)

echo_01 = qtb.EchoScan(cfg, drive_qubits, readout_resonators='R2', subspace='01', 
                       length_start=0, length_stop=400*u.us, length_points=41, level_to_fit=1)

t1s_12 = qtb.T1Scan(cfg, drive_qubits, readout_resonators='R2', subspace='12', 
                    length_start=0, length_stop=120*u.us, length_points=41, level_to_fit=2)

ramsp_12 = qtb.RamseyScan(cfg, drive_qubits, readout_resonators='R2', subspace='12', 
                          length_start=0, length_stop=20*u.us, length_points=41, 
                          artificial_detuning=0.2*u.MHz, level_to_fit=2)

echo_12 = qtb.EchoScan(cfg, drive_qubits, readout_resonators='R2', subspace='12', 
                       length_start=0, length_stop=200*u.us, length_points=41, level_to_fit=2)

t1s_23 = qtb.T1Scan(cfg, drive_qubits, readout_resonators='R2', subspace='23', 
                    length_start=0, length_stop=80*u.us, length_points=41, level_to_fit=3)

ramsp_23 = qtb.RamseyScan(cfg, drive_qubits, readout_resonators='R2', subspace='23', 
                          length_start=0, length_stop=8*u.us, length_points=41, 
                          artificial_detuning=0.5*u.MHz, level_to_fit=3)

echo_23 = qtb.EchoScan(cfg, drive_qubits, readout_resonators='R2', subspace='23', 
                       length_start=0, length_stop=160*u.us, length_points=41, level_to_fit=3)



cal_3456 = CrazyCC(cfg, drive_qubits, readout_resonators='R4', 
                   level_start=3, level_stop=6, n_seqloops=5000, save_cfg=True)
cal_3456.tones.remove('Q2/56')
cal_3456.tones.append('Q4/01')
cal_3456.rest_tones.remove('Q2/56')
cal_3456.rest_tones.append('Q4/01')

t1s_34 = qtb.T1Scan(cfg, drive_qubits, readout_resonators='R4', subspace='34', 
                    length_start=0, length_stop=60*u.us, length_points=41, level_to_fit=4)

ramsp_34 = qtb.RamseyScan(cfg, drive_qubits, readout_resonators='R4', subspace='34', 
                          length_start=0, length_stop=8*u.us, length_points=41, 
                          artificial_detuning=0.5*u.MHz, level_to_fit=4)

echo_34 = qtb.EchoScan(cfg, drive_qubits, readout_resonators='R4', subspace='34', 
                       length_start=0, length_stop=160*u.us, length_points=41, level_to_fit=4)

t1s_45 = qtb.T1Scan(cfg, drive_qubits, readout_resonators='R4', subspace='45', 
                    length_start=0, length_stop=60*u.us, length_points=41, level_to_fit=5)

ramsp_45 = qtb.RamseyScan(cfg, drive_qubits, readout_resonators='R4', subspace='45', 
                          length_start=0, length_stop=4*u.us, length_points=41, 
                          artificial_detuning=1*u.MHz, level_to_fit=5)

echo_45 = qtb.EchoScan(cfg, drive_qubits, readout_resonators='R4', subspace='45', 
                       length_start=0, length_stop=80*u.us, length_points=41, level_to_fit=5)

t1s_56 = qtb.T1Scan(cfg, drive_qubits=['Q2', 'Q4'], readout_resonators='R4', subspace=['45', '01'], 
                    length_start=0, length_stop=40*u.us, length_points=41, level_to_fit=6, 
                    main_tones=['Q4/01'], pre_gate={'Q2': [f'X180_{i}{i+1}' for i in range(5)]})

ramsp_56 = qtb.RamseyScan(cfg, drive_qubits=['Q2', 'Q4'], readout_resonators='R4', subspace=['45', '01'], 
                          length_start=0, length_stop=2*u.us, length_points=41, 
                          artificial_detuning=2*u.MHz, level_to_fit=6, 
                          main_tones=['Q4/01'], pre_gate={'Q2': [f'X180_{i}{i+1}' for i in range(5)]})

echo_56 = qtb.EchoScan(cfg, drive_qubits=['Q2', 'Q4'], readout_resonators='R4', subspace=['45', '01'], 
                       length_start=0, length_stop=60*u.us, length_points=41, level_to_fit=6, 
                       main_tones=['Q4/01'], pre_gate={'Q2': [f'X180_{i}{i+1}' for i in range(5)]})


scan_list = [
    cal_0123,
    t1s_01, ramsp_01, echo_01,
    t1s_12, ramsp_12, echo_12,
    t1s_23, ramsp_23, echo_23,
    cal_3456,
    t1s_34, ramsp_34, echo_34,
    t1s_45, ramsp_45, echo_45,
    t1s_56, ramsp_56, echo_56,
]


for i in range(total_repetition):
    for scan in scan_list:
        scan.run(f'Rep_{i}')
        plt.close('all')
        del scan.measurement
        del scan.measurements[0]

    gc.collect()
    mem = psutil.virtual_memory()

    with open('mem_record.txt', 'a') as file:
        file.write(f'{str(datetime.now())}, Round {i}, {mem[3]/1e9}GB, {mem[2]}% \n')
        
    print(f'Memory Usage: {mem[3]/1e9}GB, {mem[2]}%')
    print(f'Round {i} finished.')
    print('=' * 50)



#%% Overnight spectroscopy
# This example is from 20230702 for coarsely finding all qubit frequency on QUDIT of design round 2.
# We use ChevronScan as spectroscopy where we sweep over different drive_length and drive frequency.
# Here the four qubit in yamls actually represent four drive lines.
# We need to get some resonable readout frequency from VNA before running it.
# We don't need to do mixer correction since it's not for precise calibration.


cfg.load()
lo_freqs = (5.7*u.GHz, 5.3*u.GHz, 4.9*u.GHz, 4.5*u.GHz)

for q in range(4):
    qubit = f'Q{q}'

    for lo_freq in lo_freqs:
        cfg[f'variables.{qubit}/qubit_LO'] = lo_freq
        cfg[f'variables.{qubit}/01/freq'] = lo_freq - 250 * u.MHz
        cfg.save()
        cfg.load()

        chev = qtb.ChevronScan(cfg, drive_qubits=qubit, readout_resonators=[f'R{r}' for r in range(4)], 
                                detuning_start=-240*u.MHz, detuning_stop=240*u.MHz, detuning_points=121, 
                                length_start=0, length_stop=320*u.ns, length_points=81, n_seqloops=6)
        chev.run(f'{qubit}DriveLine_LO{lo_freq / u.GHz : 0.2g}GHz', n_pyloops=80)

        plt.close('all')
        gc.collect()
        mem = psutil.virtual_memory()
        with open('mem_record_chev.txt', 'a') as file:
            file.write(f'{str(datetime.now())}, {mem[3]/1e9}GB, {mem[2]}% \n')
        print('=' * 50)

