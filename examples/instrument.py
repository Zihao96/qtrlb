#%% Introduction

"""
Welcome to Qtrlb.
This example shows some basic usage of Qblox instrument.
For more details, please see https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/
"""

#%% Initialization

import qtrlb as qtb
from qblox_instruments import Cluster, PlugAndPlay

working_dir = r'C:\Users\machie\Box\Blok-Lab-Server\Projects\qtrlb_working_dir'
cfg = qtb.begin_measurement_session(working_dir=working_dir, variable_suffix='QUDIT', test_mode=False)
qblox = cfg.DAC.qblox


#%% To reset Qblox-Cluster instrument

qblox.stop_sequencer()
qblox.reset()

#%% To check error/problem

qblox.get_system_state()

#%% To list all the instrument connected

with PlugAndPlay() as p:
    p.print_devices()

#%% To close ethernet connection

Cluster.close_all()

#%% To reboot instrument

# This is not equivalent to switch it off and on through the hard switch on the back of instrument.
# Using PlugAndPlay, some of the chips won't really reboot or lose power during reboot. 
with PlugAndPlay() as p:
    p.reboot_all()