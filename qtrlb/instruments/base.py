import pyvisa
import serial
from typing import Any
from abc import ABCMeta, abstractmethod




class BaseInstrument(metaclass=ABCMeta):
    """
    A general structure for a python driver of an instruments.
    """
    @abstractmethod
    def __init__(self):
        return
    

    @abstractmethod
    def set(self, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        Setting parameters and sending commands to the instrument.
        """
        return
    

    @abstractmethod
    def get(self, key: str, *args: tuple[str]) -> str:
        """
        Sending commands to the instrument and getting parameters.
        """
        return
    

class VISAInstrument(BaseInstrument):
    """
    Base class for instruments connected through VISA interface.
    """
    def __init__(self, ip_address: str, **kwargs):
        self.ip_address = ip_address
        self.inst = pyvisa.ResourceManager().open_resource(f'TCPIP0::{self.ip_address}::inst0::INSTR', **kwargs)


    def set(self, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        Set the value of the given setting parameter (key) to instrument.
        """
        message = ' '.join([f'{self.command_dict[key]}', str(value), *args])
        self.inst.write(message)


    def get(self, key: str, *args: tuple[str]) -> str:
        """
        Get the value (in string) of the given setting parameter (key) from instrument.
        """
        message = ' '.join([f'{self.command_dict[key]}?', *args])
        return self.inst.query(message)
    

class SerialInstrument(BaseInstrument):
    """
    Base class for instruments connected through serial port.
    """
    def __init__(self, port: str, boudrate: int = 115200, **kwargs):
        self.port = port
        self.boudrate = boudrate
        self.inst = serial.Serial(self.port, self.boudrate, **kwargs)


    def set(self, key: str, value: Any = '', *args: tuple[str]) -> None:
        """
        Set the value of the given setting parameter (key) to instrument.
        """
        message = (' '.join([f'{self.command_dict[key]}', str(value), *args]) + '\n').encode('utf-8')
        self.inst.write(message)
      
    
    def get(self, key: str, *args: tuple[str]) -> str:
        """
        Get the value (in string) of the given setting parameter (key) from instrument.
        """
        message = (' '.join([f'{self.command_dict[key]}', *args]) + '?\n').encode('utf-8')
        self.inst.write(message)
        return self.inst.readline().decode('utf-8')

