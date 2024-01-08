# =============================================================================
# Note from Zihao (07/18/2023):
# This file includes some code that is hard to classified.
# There is also some code that is not ready for integration.
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
import qtrlb.utils.units as u
from scipy.stats import norm
from collections.abc import Iterable
from qtrlb.processing.fitting import gaussian1d_func


def compare_dict(dict_raw: dict, dict_template: dict, key: str = ''):
    """
    Recursively checking the keys of two dictionaries without raising exception.
    Suffix tells which yaml we are checking.
    """
    for k, v in dict_template.items():
        if not k in dict_raw: print(f'misc: Missing key "{k}" in key "{key}".')
        if isinstance(v, dict): 
            compare_dict(dict_raw[k], dict_template[k], k)


def tone_to_qudit(tone: str | Iterable) -> str | list:
    """
    Translate the tone to qudit.
    Accept a string or a list of string.

    Example:
    tone_to_qudit('Q2') -> 'Q2'
    tone_to_qudit('R2') -> 'R2'
    tone_to_qudit('Q2/12') -> 'Q2'
    tone_to_qudit(['Q2/01', 'Q2/12', 'R2/a']) -> ['Q2', 'R2']
    tone_to_qudit([['Q2/01', 'Q3/12', 'R2/a'], ['Q2/01', 'Q2/12', 'R2']]) -> [['Q2', 'Q3', 'R2'], ['Q2', 'R2']]
    """
    if isinstance(tone, str):
        assert tone.startswith(('Q', 'R')), f'Cannot translate {tone} to qudit.'
        try:
            qudit, _ = tone.split('/')
            return qudit
        except ValueError:
            return tone
        
    elif isinstance(tone, Iterable):
        qudit = []
        for t in tone:
            q = tone_to_qudit(t)
            if q not in qudit: qudit.append(q)
        return qudit

    else:
        raise TypeError(f'misc: Cannot translate the {tone}. Please check it type.')


def find_subtones(qudit: str, tones: Iterable) -> list:
    """
    Find all subtones in a tones list that is associated to a qudit string.
    If the tones list has resonators that are not subtones, it won't be added.

    Example:
    tones = ['Q4/01', 'Q4/12', 'Q5/01', 'Q5/12', 'R4/a', 'R4/b', 'R4/c', 'R4', 'R5/a', 'R5/b', 'R5']
    find_subtones('Q4', tones) -> ['Q4/01', 'Q4/12']
    find_subtones('R4', tones) -> ['R4/a', 'R4/b', 'R4/c']
    """
    return [tone for tone in tones if tone.startswith(qudit) and tone != qudit]


def split_subspace(subspace: str) -> tuple[int, int]:
    """
    Given a string like '23', '02', '910', '1011', split into two integer.
    """
    assert subspace.isdecimal(), f'misc: Subspace string must be decimal.'
    level_low = int( subspace[ : int(len(subspace)/2)] )
    level_high = int( subspace[ int(len(subspace)/2) : ] )
    return level_low, level_high

