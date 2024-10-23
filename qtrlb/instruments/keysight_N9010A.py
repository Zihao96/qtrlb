"""
This file was originally written by Maddy Ramsey when she was a REU student in BlokLab in 2023 summer.
I rewrite it later but keep her idea and main effort on construct SCPI command.
"""
import numpy as np
from typing import Any
from qtrlb.instruments.base import VISAInstrument




class Keysight_N9010A(VISAInstrument):
    """ Python driver for Keysight EXA Signal Analyzer N9010A.
    """

    command_dict = { 
        'freq_span': 'FREQuency:SPAN',
        'freq_span_prev': 'FREQuency:SPAN:PREVious',
        'freq_center': 'FREQuency:CENTer',
        'res_bw': 'BANDwidth:RESolution',
        'vid_bw': 'BANDwidth:VID',
        'vid_bw_auto': 'BAND:VID:AUTO',
        'peak_list': 'CALC:DATA:PEAK',
        'data': 'CALCulate:DATA',
        'ref_level': ':DISPlay:WINDow:TRACe:Y:RLEVel',
        'scale/div': ':DISPlay:WINDow:TRACe:Y:PDIVision',
        'excursion': ':CALCulate:MARKer:PEAK:EXCursion',
        'excursion_status': ':CALCulate:MARKer:PEAK:EXCursion:STATe',
        'threshold': ':CALCulate:MARKer:PEAK:THReshold',
        'threshold_status': ':CALCulate:MARKer:PEAK:THReshold:STATe',
        'continuous_sweep':'INIT:CONT',
        'sweep_points': ':SWEep:POINts'
    }    

    marker_command_dict = {
        'max_peak': ':MAXimum',
        'next_peak': 'MAXimum:NEXT',
        'peak_right': ':MAXimum:RIGHt',
        'peak_left': ':MAXimum:LEFT',
        'x': ':X',
        'y': ':Y',
        'freq': ':X',
        'ON': ':MODE POS',
        'OFF': ':MODE OFF',
        'all_off': ':AOFF'
    }


    def __init__(self, ip_address: str = '192.168.1.14', **kwargs):
        super().__init__(ip_address, **kwargs)


    def get_marker(self, marker: int | str, key: str, *args: tuple[str]) -> str:
        """
        Get the value of the given marker parameter (key) from instrument.
        """
        message = ' '.join([f'CALCulate:MARKer{marker}{self.marker_command_dict[key]}?', *args])
        return self.inst.query(message)


    def set_marker(self, marker: int | str, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        Set the value of the given marker parameter (key) from instrument.
        """
        message = ' '.join([f'CALCulate:MARKer{marker}{self.marker_command_dict[key]}', str(value), *args])
        self.inst.write(message)
    

    def set_marker_center(self, marker: int | str):
        """
        Convenient function to set a given maker to the center frequency.
        """
        center_freq = self.get('freq_center')
        self.set_marker(marker, 'x', value=center_freq)


    @property
    def data(self):
        """
        Date acquired by the instrument. x is frequency, y is power.
        """
        data_str = self.get('data')
        val_array = np.genfromtxt([data_str], delimiter=",")
        x, y = val_array[0::2], val_array[1::2]
        return x, y

