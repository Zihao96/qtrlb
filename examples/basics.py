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

# Our RabiScan will keep freq/amp while changing drive pulse length before readout.
# The length we will scan over will be np.linspace(start, stop, points).
# Here the length is the only variables and we call it a 1D Scan where length is our x-axis.
# In fact, for universality, the attribute name is rs.x_start, rs.x_stop, rs.x_points, rs.x_values.

# The subspace specify which subspace the experiment is working on. 
# A 'X180_01' gate will be added to beginning automatically.
# The level_to_fit should be one of the readout levels associated to a readout_resonators.
# If we don't use classification, then 0 is I, 1 is Q.

# The experiment here will repeat 2000 times.
# The iteration of n_seqloops is implemented in Q1ASM sequence program, not python.
# However, one sequence can acquire at most 131072 readout data on each I-Q path.
# Here, our readout points will be 51 * 2000 = 102000.
# If we need to do more repetition, we can specify n_pyloops in run() method.
# n_pyloops is implemented in python and slightly slower.
# In that case, the total repetition will be n_seqloops * n_pyloops.



#%% Mulitplexing drive/readout

rs = qtb.RabiScan(cfg, drive_qubits=['Q2', 'Q4'], readout_resonators=['R2', 'R4'])
rs.run('example_multiplexing')

# To run multi-qubit/resonator experiment, we can pass a list to these arguments.
# In fact, everything in Scan is either list of dict.
# It's for multiplexing and also to make a synchronized quantum circuit.
# In this example, the two rabi drive with come out at same time with their own freq/amp.
# And our readout will have two I+jQ pair for each single shot.