#%% Introduction

"""
Welcome to Qtrlb.
This example shows some basic usage of Qtrlb.
We will create a MetaManager called cfg and run a Rabi experiment with it.
"""

#%% Initialize the cfg(MetaManager), no need for setting console working directory

import qtrlb as qtb
import qtrlb.utils.units as u

# We expect a Yamls folder under this path and yamls files living inside it.
working_dir = r'C:\Users\machie\Box\Blok-Lab-Server\Projects\qtrlb_working_dir'

cfg = qtb.begin_measurement_session(working_dir=working_dir, variable_suffix='QUDIT', test_mode=False)
qblox = cfg.DAC.qblox


#%% Define useful variables

drive_qubits = 'Q2'
readout_resonators = 'R2'
subspace = '01'
level_to_fit = 1


#%% Run Rabi experiment

rs = qtb.RabiScan(cfg, drive_qubits, readout_resonators)
rs.run('example')

# When it finished, a plot will show in Jupyter console.
# The plot, data and configuration will be saved to rs.data_path.
# The specific path will be determined by the value we set in data.yaml
# We can also access the result through rs.measurement, rs.fit_result, rs.figures. All dict.


#%% To repeat experiment with same setup, we can simple call run again

rs.run('example_again')

# This will create a new folder, and new result will be saved into it.
# After these, rs.measurements, which is a list, will have two element in it.
# Each element is a dict, with resonator names as keys.


#%% Specify other parameters

# Here we use "u" to make unit easier to write and read.
rs = qtb.RabiScan(cfg, drive_qubits, readout_resonators,
                  length_start=0*u.ns, length_stop=600*u.ns, length_points=51,
                  subspace='12', level_to_fit=1, n_seqloops=2000)
rs.run('example_MoreParam')


# TODO: what are these, the 131072, they will become attribute but x/y instead of length/amp.





#%% Mulitplexing drive/readout

rs = qtb.RabiScan(cfg, drive_qubits=['Q2', 'Q4'], readout_resonators=['R2', 'R4'])
rs.run('example_multiplexing')

# TODO: Also tell user everything is list/dict here, the subspace/level_to_fit should map each other, and main_tone.