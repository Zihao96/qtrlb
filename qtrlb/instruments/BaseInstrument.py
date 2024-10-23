from typing import Any
from abc import ABCMeta, abstractmethod
import pyvisa
import serial
import numpy as np


class BaseInstrument(metaclass = ABCMeta):
    """
    A Python parent class for instruments comunicated via either visa or serial 
    """
    @abstractmethod
    def __init__(self):
        return
    

    @abstractmethod
    def set(self, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        A general form of the setters to be overwritten for every child class.
        """
        return
    

    @abstractmethod
    def get(self, key: str, *args):
        return
    

class VisaInstrument(BaseInstrument):
    """
    a child class of Instrument Base specific for visa instruments
    """
    def __init__(self, ip_address: str):
        self.ip_address = ip_address
        self.inst = pyvisa.ResourceManager().open_resource(f'TCPIP0::{self.ip_address}::inst0::INSTR')


    def set(self, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        Set the value of the given setting parameter (key) to instrument.
        Normally it will return to the str, unless we try to get data.
        """
        message = ' '.join([f'{self.command_dict[key]}', str(value), *args])
        self.inst.write(message)


    def get(self, key: str, *args: tuple[str]) -> str:
        """
        Get the value of the given setting parameter (key) from instrument.
        Normally it will return to the str, unless we try to get data.
        """
        message = ' '.join([f'{self.command_dict[key]}?', *args])
        return self.inst.query(message)
    

class SerialInstrument(BaseInstrument):
    """
    a child class of Instrument Base specific for Serial instruments
    """
    def __init__(self, port: str, boudrate: int = 115200, **kwargs):
        self.port = port
        self.boudrate = boudrate
        self.inst = serial.Serial(self.port, self.boudrate, **kwargs)


    def set(self, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        Set the value of the given setting parameter (key) to instrument.
        Normally it will return to the str, unless we try to get data.
        """
        message_temp = (' '.join([f'{self.command_dict[key]}', str(value), *args]))
        message = (message_temp + '\n').encode('utf-8')
        self.inst.write(message)
      
    
    def get(self, key: str, *args: tuple[str]) -> str:
        """
        Get the value of the given setting parameter (key) from instrument.
        Normally it will return to the str, unless we try to get data.
        """
        message_temp = (' '.join([f'{self.command_dict[key]}', *args]))
        message = (message_temp + '?\n').encode('utf-8')
        self.inst.write(message)
        return self.inst.readline().decode('utf-8')

