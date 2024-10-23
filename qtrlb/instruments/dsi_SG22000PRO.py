from qtrlb.instruments.BaseInstrument import SerialInstrument


class DSISignalGenerator_SG22000PRO(SerialInstrument):
    """ Python driver for DS Instrument Signal Generator SG22000PRO.
    """

    command_dict = {
        'idn': '*IDN',
        'ping': '*PING',
        'system_error': 'SYST:ERR',
        'clear_error': '*CLS',
        'hardware_revision_number': '*REV',
        'reset': '*RST',
        'opc': '*OPC',
        'display': '*DISPLAY',
        'save_state': '*SAVESTATE',
        'clear_state': '*CLEARSTATE',
        'device_name': '*UNITNAME',
        'temperature': '*TEMPC',
        'freq_min': 'FREQ:MIN',
        'freq_max': 'FREQ:MAX',
        'freq': 'FREQ:CW',
        'output': 'OUTP:STAT',
        'power': 'POWER',
    }


    @property
    def completed(self):
        """
        Check whether the opeation has completed.
        """
        state = True if self.get('opc') == '+1\r\n' else False
        return state
