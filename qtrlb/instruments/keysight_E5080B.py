import numpy as np
from typing import Any
from qtrlb.instruments.base import VISAInstrument



    
class Keysight_E5080B(VISAInstrument):
    """ Python driver for Keysight ENA VNA E5080B.
    """

    command_dict = {
        # General command
        'idn': "*IDN",
        'opc': "*OPC",
        'preset': "SYSTem:FPReset",
        'create_display':'DISP:WIND:STAT',
        'create_measurment': 'CALC:MEAS:DEF',
        'display_measurment':'DISP:MEAS:FEED',
        'read_measurment':'CALC:PAR:CAT:EXT',
        'read_window': 'DISP:CAT',
        'measurment_title':'DISP:MEAS1:TITL',

        # Measurement command
        'format': 'CALC:MEAS:FORM',  # MLOG, MLIN, PHASE, POLAR, SMITH
        'freq_start':'SENS:FREQ:STAR',
        'freq_stop':'SENS:FREQ:STOP',
        'freq_span':'SENS:FREQ:SPAN',
        'freq_center':'SENS:FREQ:CENT',
        'power': 'SOUR:POW',
        'avg_enable':'SENS:AVER',
        'n_avg':'SENS:AVER:COUN',
        'clear_avg':'SENS:AVER:CLE',
        'avg_state':'STAT:OPER:AVER:COND', # Returns a string of "0" for logical false, or else for logical true
        'n_points': 'SENS:SWE:POIN',
        'electrical_delay': 'CALC:MEAS:CORR:EDEL',
        'y_scale_auto':'DISP:MEAS:Y:AUTO',
        'y_scale/div':'	DISP:MEAS:Y:PDIV',
        'y_scale_position':'DISP:MEAS:Y:RPOS',
        'y_ref_level':'DISP:MEAS:Y:RLEV',
        'y_spacing':'DISP:WIND:TRAC:Y:SPAC',
        'sweep_mode':'SENS:SWE:MODE',
        'sweep_time':'SENS:SWE:TIME',
        'x_data': "CALC:MEAS:X",
        'y_data': "CALC:MEAS:DATA:FDATA",
        'IQ_data': "CALC:MEAS:DATA:SDATA",
        'res_bw': 'SENS:SA:BANDwidth:RESolution',
        'vid_bw': 'SENS:SA:BANDwidth:VID',
        'vid_bw_auto': 'SENS:SA:BAND:VID:AUTO',
    }

    marker_command_dict = {
        'x':':X',
        'on_off':':REF:STAT'
    }


    def get_marker(self, marker: int | str, key: str, *args: tuple[str]) -> str:
        """
        Get the value of the given marker parameter (key) from instrument.
        """
        message = ' '.join([f'CALC:MEAS:MARK{marker}{self.marker_command_dict[key]}?', *args])
        return self.inst.query(message)


    def set_marker(self, marker: int | str, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        Set the value of the given marker parameter (key) from instrument.
        """
        message = ' '.join([f'CALC:MEAS:MARK{marker}{self.marker_command_dict[key]}', str(value), *args])
        self.inst.write(message)


    @property
    def avg_completed(self):
        """
        Check whether the averaging has completed.
        """
        state = False if self.get('avg_state') == '+0\n' else True
        return state
    

    def create_measurement(self, format: str = 'S21'):
        """
        Create a measurement.
        """
        self.set('preset')
        self.set('create_display', '1')
        self.set('create_measurment', f"'{format}'")
        self.set('display_measurment', '1')
            

    @property
    def x_data(self):
        """
        Get x data.
        """
        data_str = self.get('x_data')
        return np.genfromtxt([data_str], delimiter=",")


    @property
    def y_data(self):
        """
        Get y data.
        """
        data_str = self.get('y_data')
        return np.genfromtxt([data_str], delimiter=",")

    
    @property
    def IQ_data(self):
        """
        Get raw IQ data without the correction for electrical delay.
        """
        data_str = self.get('IQ_data')
        val_array = np.genfromtxt([data_str], delimiter=",")
        I, Q = val_array[0::2], val_array[1::2]
        return I, Q

