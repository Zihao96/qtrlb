"""
This file was originally written by Maddy Ramsey when she was a REU student in BlokLab in 2023 summer.
I rewrite it later but keep her idea and main effort on construct SCPI command.
"""


import pyvisa
import numpy as np
from typing import Any


class N9010A:
    """ Python driver for Keysight EXA Signal Analyzer N9010A.
        It's designed to automate mixer correction.
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

    def __init__(self, ip_address: str = '192.168.1.14'):
        self.ip_address = ip_address
        self.resource_name = f'TCPIP0::{self.ip_address}::inst0::INSTR'
        self.inst = pyvisa.ResourceManager().open_resource(self.resource_name)


    def get(self, key: str, *args: tuple[str]) -> str | np.ndarray:
        """
        Get the value of the given setting parameter (key) from instrument.
        Normally it will return to the str, unless we try to get data.
        """
        message = ' '.join([f'{self.command_dict[key]}?', *args])
        return self.inst.query(message)


    def set(self, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        Set the value of the given setting parameter (key) to instrument.
        Normally it will return to the str, unless we try to get data.
        """
        message = ' '.join([f'{self.command_dict[key]}', str(value), *args])
        self.inst.write(message)


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
        data_str = self.get('data')
        val_array = np.genfromtxt([data_str], delimiter=",")
        x_vals, y_vals = val_array[0::2], val_array[1::2]
        return x_vals, y_vals

