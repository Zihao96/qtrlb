import numpy as np
import pandas as pd
from copy import deepcopy
from warnings import simplefilter

PI = np.pi
# Disable a possible warning message from pandas.
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)




def dict_to_DataFrame(dic: dict, name: str, rows: list, padding: object = 'I'):
    """
    Turn a dictionary into a Pandas DataFrame with padding.
    Each key in dic or element in rows will become index (row) of the DataFrame.
    Each column will be renamed as 'name_0', 'name_1'.
    
    Example:
        dict: {'Q3':['X180_01', 'X180_12'], 'Q4':['Y90_01']}
        name: 'pregate'
        rows: ['Q3', 'Q4', 'R3', 'R4']
    """
    dic = deepcopy(dic)
    for row in rows:
        if row not in dic: dic[row] = []
        
    dataframe = pd.DataFrame.from_dict(dic, orient='index')
    dataframe = dataframe.rename(columns={i:f'{name}_{i}' for i in range(dataframe.shape[1])})
    dataframe = dataframe.fillna(padding)        
    return dataframe


def gate_transpiler(gate_df: pd.DataFrame, tones: list) -> pd.DataFrame:
    """
    Take a gate pandas DataFrame and decompose it into different sequencers(tones).
    Return a pulse DataFrame whose rows are tones.
    Right now we assume pulse_df has same number of columns as gate_df where lengths can be reused.
    In future, we can pass in the length of the gate and decompose it into multiple consecutive pulse.
    Then it will return to a DataFrame with more columns, where each of them may have a individual length.

    Parameters:
        gate_df: A pandas DataFrame, see example below.
        tones: A list of the row for the returned DataFrame.

    Example:
    gate_df = 
           subspace_0 subspace_1 prepulse_0 prepulse_1 Readout
        Q3    X180_01          I     Y90_01          I       I
        Q4    X180_01    X180_12         H3     Z90_12       I
        R3          I          I          I          I  RO_a_b
        R4          I          I          I          I    RO_a

    With tones = ['Q3/01', 'Q3/12', 'Q4/01', 'Q4/12', 'R3/a', 'R3/b', 'R4/a'], this function return it to

              subspace_0 subspace_1 prepulse_0 prepulse_1 Readout
        Q3/01       X180          I        Y90          I       I
        Q3/12          I          I          I          I       I
        Q4/01       X180          I      H3_01          I       I
        Q4/12          I       X180      H3_12        Z90       I
        R3/a           I          I          I          I      RO
        R3/b           I          I          I          I      RO
        R4/a           I          I          I          I      RO
    """

    pulse_df = dict_to_DataFrame(dic={}, name='', rows=tones)

    # col_name is string, column is a Series.
    for col_name in gate_df:
        pulse_df[col_name] = 'I'  # Add empty column to pulse_df first. Then fill with pulse.
        column = gate_df[col_name]

        # Both row_name and row are string.
        for row_name, gate in column.items():

            if gate == 'I':
                pass

            elif gate.startswith('RO'):
                subtones = gate.split('_')
                for subtone in subtones[1:]: pulse_df.loc[f'{row_name}/{subtone}', col_name] = 'RO'

            elif gate.startswith(('X', 'Y', 'Z')):
                gate_str, subspace = gate.split('_')
                pulse_df.loc[f'{row_name}/{subspace}', col_name] = gate_str

            elif gate.startswith('H3'):
                pulse_df.loc[f'{row_name}/01', col_name] = f'{gate}_01'
                pulse_df.loc[f'{row_name}/12', col_name] = f'{gate}_12'

            elif gate.startswith('D'):
                nlevels = int(gate[1:])
                for n in range(nlevels-1):
                    subspace = f'{n}{n+1}'
                    pulse_df.loc[f'{row_name}/{subspace}', col_name] = f'{gate}_{subspace}'

            else:
                raise ValueError(f"Pulses: Gate {gate} hasn't been defined.")
            
    return pulse_df




def pulse_interpreter(cfg, tone: str, pulse_string: str, length: int, **pulse_kwargs) -> str:
    """
    Generate the string sequence program for Qblox sequencer based on a input string.
    
    Parameters:
        cfg: A Metamanager, typically belong to a Scan.
        tone: String of single qubit or resonator along with its subspace. Example: 'Q3/12', 'R4'.
        pulse_string: The content we need to interpret. Example: 'X180', 'I', 'RO', 'H3_01'.
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
        
        freq = round(cfg.variables[f'{tone}/mod_freq'] * 4)
        gain = round(cfg.variables[f'{tone}/amp'] * 32768)
        tof_ns = round(cfg.variables['common/tof'] * 1e9)
        
        pulse_program = f"""
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain}
                    reset_ph
                    play             0,0,{tof_ns} 
                    acquire          {acq_index},R1,{length - tof_ns}
        """
        
    elif pulse_string.startswith('X'):
        angle = pulse_string[1:]
        tone_dict = cfg[f'variables.{tone}']
        
        freq = round((tone_dict['mod_freq'] + tone_dict['pulse_detuning']) * 4)
        gain = calculate_angle_to_gain(angle, tone_dict['amp_180'], tone_dict['amp_90'])
        gain_drag = round(gain * tone_dict['DRAG_weight'])
        
        pulse_program = f"""
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain_drag}
                    play             0,1,{length} 
        """
        
    elif pulse_string.startswith('Y'): # TODO: fix this for negative modulation frequencies (swap
                                       # the order of the phase changes)
        # Y = Z90 * X * Z-90, operator order, so Z-90 first.
        angle = pulse_string[1:]
        tone_dict = cfg[f'variables.{tone}']
        
        freq = round((tone_dict['mod_freq'] + tone_dict['pulse_detuning']) * 4)
        gain = calculate_angle_to_gain(angle, tone_dict['amp_180'], tone_dict['amp_90'])
        gain_drag = round(gain * tone_dict['DRAG_weight'])
        
        pulse_program = f"""
                    set_ph_delta     {round(750e6)}
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain_drag}
                    play             0,1,{length}
                    set_ph_delta     {round(250e6)}
        """
        
    elif pulse_string.startswith('Z'):
        angle = pulse_string[1:]
        tone_dict = cfg[f'variables.{tone}']
        # making sure we handle negative frequencies correctly - EC 2023-11-29
        if tone_dict['mod_freq'] > 0:
            angle = round(float(angle) / 360 * 1e9)
        else:
            angle = round((1 - float(angle) / 360) * 1e9)
        
        pulse_program = f"""
                    set_ph_delta     {angle}
        """
        pulse_program += '' if length == 0 else f""" 
                    upd_param        {length}
        """
    
    elif pulse_string.startswith('H3'):
        pulse_dict = cfg[f'gates.H3:{tone}']
        subspace = tone.split('/')[-1]
        
        freq = round((pulse_dict['mod_freq'] + pulse_dict['pulse_detuning']) * 4)
        gain = pulse_dict['amp']
        gain_drag = round(gain * pulse_dict['DRAG_weight'])
        waveform_index = pulse_dict['waveform_index']
        prephase = round(pulse_dict['prephase'][subspace] / (2*PI) * 1e9)
        postphase = round(pulse_dict['postphase'][subspace] / (2*PI) * 1e9)

        pulse_program = f"""
                    set_ph_delta     {prephase}
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain_drag}
                    play             {waveform_index},{waveform_index+1},{length}
                    set_ph_delta     {postphase}
        """

    elif pulse_string.startswith('D'):
        # displacements as gates - EC 2023-11-28
        # it's a displacement of some sort. extract the dimension:
        gate_string, subspace = pulse_string.split('_')
        nlevels = int(gate_string[1:])

        # get the pulse_dict and extract the parameters
        pulse_dict = cfg[f'gates.D{nlevels}:{tone}']
        mod_freq_sign = np.sign(pulse_dict['mod_freq'])
        freq = round(4 * (pulse_dict['mod_freq'] + mod_freq_sign * pulse_dict['detuning']))
        gain = round(pulse_dict['amp_reference'] * pulse_dict['amp_scale_factor'] * 32768)
        gain_drag = -round(pulse_dict['DRAG_weight'] * gain)
        waveform_index = pulse_dict['waveform_index']

        # build the pulse_program
        pulse_program = f"""
                    set_freq         {freq}
                    set_awg_gain     {gain},{gain_drag}
                    play             {waveform_index},{waveform_index+1},{length}
        """

    else:
        raise ValueError(f'Pulses: The pulse "{pulse_string}" cannot be interpreted.')
    
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