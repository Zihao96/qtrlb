

def pulse_interpreter(cfg, qudit: str, pulse_string: str, length: int, **pulse_kwargs) -> str:
    """
    Generate the string sequence program for Qblox sequencer based on a input string.
    
    Attribute:
        cfg: A Metamanager, typically belong to a Scan.
        qudit: String of single qubit or resonator. Example: 'Q3', 'R4'.
        pulse_string: The content we need to interpret. 
                      Example: 'X180_12', 'I', 'RO', 'H_23'.
        length: In unit of [ns]. It specifies the length of pulse/program.
        pulse_kwargs: Use to specifiy the acquisition index for readout.
    """
    
    if pulse_string == 'I':
        pulse_program = '' if length == 0 else f"""
                    wait             {length}
        """
        
    elif pulse_string == 'RO':
        if 'acq_index' not in pulse_kwargs:
            print('Pulses: No acquisition index specified. Index 0 will be used')
            acq_index = 0
        else:
            acq_index = pulse_kwargs['acq_index']
        
        freq = round(cfg.variables[f'{qudit}/mod_freq'] * 4)
        gain = round(cfg.variables[f'{qudit}/amp'] * 32768)
        tof_ns = round(cfg.variables['common/tof'] * 1e9)
        
        pulse_program = f"""
                    set_freq         {freq}
                    set_awg_gain     {gain},0
                    play             0,0,{tof_ns} 
                    acquire          {acq_index},R1,{length - tof_ns}
        """
        
    elif pulse_string.startswith('X'):
        angle, subspace = pulse_string[1:].split('_')
        subspace_dict = cfg[f'variables.{qudit}/{subspace}']
        
        freq = round((subspace_dict['mod_freq'] + subspace_dict['pulse_detuning']) * 4)
        gain = calculate_angle_to_gain(angle, subspace_dict['amp_180'], subspace_dict['amp_90'])
        gain_drag = round(gain * subspace_dict['DRAG_weight'])
        
        pulse_program = f"""
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain_drag}
                    play             0,1,{length} 
        """
        
    elif pulse_string.startswith('Y'):
        angle, subspace = pulse_string[1:].split('_')
        subspace_dict = cfg[f'variables.{qudit}/{subspace}']
        
        freq = round((subspace_dict['mod_freq'] + subspace_dict['pulse_detuning']) * 4)
        gain = calculate_angle_to_gain(angle, subspace_dict['amp_180'], subspace_dict['amp_90'])
        gain_drag = round(gain * subspace_dict['DRAG_weight'])
        
        pulse_program = f"""
                    set_ph_delta     {round(250e6)}
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain_drag}
                    play             0,1,{length} 
                    set_ph_delta     {round(750e6)}
        """
        
    elif pulse_string.startswith('Z'):
        angle, subspace = pulse_string[1:].split('_')
        angle = round(float(angle) / 360 * 1e9)
        
        pulse_program = f"""
                    set_ph_delta     {angle}
        """
        pulse_program += '' if length == 0 else f""" 
                    wait             {length}
        """
        
    
    elif pulse_string.startswith('H'):
        # H = Y90 * Z, in operator order, so Z first.
        _, subspace = pulse_string.split('_')
        subspace_dict = cfg[f'variables.{qudit}/{subspace}']
        freq = round((subspace_dict['mod_freq'] + subspace_dict['pulse_detuning']) * 4)
        gain = round(subspace_dict['amp_90'] * 32768)
        gain_drag = round(gain * subspace_dict['DRAG_weight'])
        
        pulse_program = f"""
                    set_ph_delta     {round(750e6)}
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain_drag}
                    play             0,1,{length} 
                    set_ph_delta     {round(750e6)}
        """
    
    else:
        raise ValueError(f'The pulse "{pulse_string}" cannot be interpreted.')
    
    return pulse_program


def calculate_angle_to_gain(angle: str | float, amp_180: float, amp_90: float) -> int:
    """
    Calculate the gain value for rotation angle other than 180 and 90.
    Support negative angle.
    
    Note from Zihao(2023/03/14):
    We assume this relation is linear here, which is not very precise. 
    It's not only because Bloch sphere is not sphere, 
    but also coming from the nonlinearity of the instrument.
    Experiment on Tektronix gives deviation from linearity below 0.5%.
    """
    angle = float(angle)
    
    if abs(angle) <= 90:
        amp_angle = amp_90 * angle / 90
    else:
        amp_angle = amp_180 * angle / 180
        
    return round(amp_angle * 32768)