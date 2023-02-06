


def pulse_interpreter(config, qudit: str, pulse_string: str, length: int):
    """
    Generate the string sequence program for Qblox sequencer based on a input string.
    
    Attribute:
        config: A Metamanager, typically belong to a Scan.
        qudit: String of single qubit or resonator. Example: 'Q3', 'R4'.
        pulse_string: The content we need to interpret. Example: 'X180_12', 'I', 'Readout'.
        length: In unito of [ns]. It used for Identity to specify how long we wait.
    """
    
    if pulse_string == 'I':
        pulse_program = f"""
                    wait             {length}            # Identity.
        """
        
    elif pulse_string == 'Readout':
        pass
        
        
    elif pulse_string.startswith('X'):
        angle, subspace = pulse_string[1:].split('_')
        freq = int(config.varman[f'{qudit}/{subspace}/mod_freq']*4)
        gain = int(config.varman[f'{qudit}/{subspace}/amp_{angle}'] * 32768)
        pulse_program = f"""
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain}
                    play             0,0,{length} 
        """
        
        
    elif pulse_string.startswith('Y'):
        
    elif pulse_string.startswith('Z'):
    
    elif pulse_string.startswith('H'):
        
    # TODO: Write it.
    elif pulse_string.startswith('CR'):
        pulse_program = ''
    
    return pulse_program