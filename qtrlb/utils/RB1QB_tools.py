#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:38:29 2023

@author: Z

I'm very sorry to tell you if you want to implement RB on higher subspace than\
'01', the best way is to replace all '01' in these page to the target subspace.
I plan to make it better after I map sequencer to each subspace, which is hard.
"""

import secrets
import numpy as np
from copy import deepcopy
from qiskit import QuantumCircuit
from qiskit import BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator
from cirq.linalg.predicates import allclose_up_to_global_phase
PI = np.pi


def unitary(theta: float, axis: tuple, eliminate_float_error: bool = True) -> np.ndarray:
    """
    Calculate the 2x2 unitary given a qubit rotation angle and axis.
    Reference: Nilsen and Chuang Eq.(4.8)
    """
    axis_unit = np.array(axis) / np.linalg.norm(axis)
    U = np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) \
        * (  axis_unit[0] * np.array( ((0,1),(1,0)) ) 
           + axis_unit[1] * np.array( ((0,-1j),(1j,0)) ) 
           + axis_unit[2] * np.array( ((1,0),(0,-1)) ) )
        
    if eliminate_float_error: U = np.around(U, decimals=15) 
    return U
    
    
def transpile_unitary_to_circuit(U: np.ndarray, 
                                 basis_gates: tuple = ('x', 'z', 'rx', 'rz')) -> None:
    """
    Using Qiskit transpile function to decompose a unitary into primitive gate.
    It may fail if the primitive gate set is not universal. 
    Reference:
    https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html
    https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html
    """
    backend = BasicAer.get_backend('qasm_simulator')
    oper = Operator(U)
    circ = QuantumCircuit(1)
    circ.append(oper, [0])
    circ = transpile(circ, backend, basis_gates=list(basis_gates), optimization_level=3)
    return circ
    
    
def calculate_combined_unitary(U_list: list | np.ndarray,
                               eliminate_float_error: bool = True) -> np.ndarray:
    """
    Expect to get a list of unitary (ndarray) in time order (left one happen first).
    Using numpy.matmul to calculate the result.
    The unitaries should have correct shape for doing multiplication.
    """
    # Here I use size of last index of first matrix to generate a identity.
    result = np.eye(U_list[0].shape[-1])
    for U in U_list:
        result = U @ result
    
    if eliminate_float_error: result = np.around(result, decimals=15) 
    return result


def calculate_combined_operator(U_list: list | np.ndarray) -> Operator:
    """
    Expect to get a list of unitary (ndarray) in time order (left one happen first).
    Using qiskit unitary simulator to calculate the result.
    The unitaries should have correct shape for doing multiplication.
    """
    circ = QuantumCircuit(1)
    for U in U_list:
        circ.append(Operator(U), [0])
    
    return Operator(circ)


def find_Clifford_gate(U: np.ndarray, Clifford_gates: dict) -> str:
    """
    Given an expression of unitary, find the name of its corresponding Clifford gate.
    Allow a difference with global phase.
    """
    for k, v in Clifford_gates.items():
        if allclose_up_to_global_phase(v['unitary'], U): return k
    
    raise ValueError('There is no such Cliford gate!')
    
    
def generate_RB_Clifford_sequences(Clifford_gates: dict, n_gates: int, 
                                   n_random: int = 30) -> list:
    """
    Generate n_random different RB sequence where each sequence has n_gates Clifford gates.
    Return to a ndarray with shape (n_random, n_gates+1), and each entry is a string.
    Notice the dictionary Clifford_gates will be changed in situ.
    """
    for v in Clifford_gates.values():
        if 'unitary' in v: continue
        v['unitary'] = unitary(v['theta'], v['axis'])
        
    if n_gates == 0: return [ [] for i in range(n_random) ]
        
    Clifford_sequences = np.zeros(shape=(n_random, n_gates+1), dtype='U7')
    Clifford_sequences_mat = np.zeros(shape=(n_random, n_gates, 2, 2), dtype='complex128')
    
    for i in range(n_random):
        for j in range(n_gates):
            key_random = secrets.choice(list(Clifford_gates.keys()))
            Clifford_sequences[i, j] = key_random
            Clifford_sequences_mat[i, j] = Clifford_gates[key_random]['unitary']
            
        total_U = calculate_combined_unitary(Clifford_sequences_mat[i])
        inverse = find_Clifford_gate(total_U.T.conj(), Clifford_gates)
        Clifford_sequences[i, -1] = inverse
        
    return Clifford_sequences.tolist()


def generate_RB_primitive_sequences(Clifford_sequences: list,
                                    Clifford_to_primitive: dict) -> list:
    """
    Transpile all Clifford gate in sequences into primitive_gate.
    Return list since we can't guarantee the length are same for all sequences.
    """
    sequences = deepcopy(Clifford_sequences)
    for i, seq in enumerate(sequences):
        sequences[i] = []
        for Clifford in seq:
            sequences[i] += Clifford_to_primitive[Clifford]
            # Iterable unpacking cannot be used in comprehension.
            
        # A temporary code to simplify circuit for qblox.
        sequences[i] = optimize_circuit(sequences[i])
        
    return sequences
    

def optimize_circuit(sequence: list) -> None:
    """
    Remove all Identity and combine all adjacent Z gates.
    I'm sorry. Hopefully this is fast enough.
    """
    sequence = [gate for gate in sequence if gate != 'I']
    optimized_sequence = []
    
    i = 0
    while i < len(sequence):
        if sequence[i].startswith('X'): 
            optimized_sequence.append(sequence[i])
            i += 1
            
        elif sequence[i].startswith('Z'):
            
            for j, sub_gate in enumerate(sequence[i:]):
                if sub_gate.startswith('X'): break
            
            if j <= 1: 
                optimized_sequence.append(sequence[i])
                i += 1
            else:
                optimized_sequence.append( combine_Z_gates(sequence[i:i+j]) ) 
                i += j
            
    optimized_sequence = [gate for gate in optimized_sequence if gate != 'I']
    return optimized_sequence
    

def combine_Z_gates(sequence: list) -> str:
    """
    Combine a list of Z gates into one single Z gate.
    """
    angle = 0
    for gate in sequence:
        angle += int(gate.split('_')[0][1:]) 
        
    angle = angle % 360
    return f'Z{angle}_01' if angle != 0 else 'I'
    



    
#%% Definition of single qubit Clifford.
# Notice the definition of 'h' may not be what you expect as Y90.

primitive_gates = ['X180_01', 'X90_01', 'X-90_01',
                   'Z180_01', 'Z90_01', 'Z270_01', 'I']

Clifford_gates = {
    'I': {'theta': 0, 'axis': (0, 0, 1)}, 
    'X': {'theta': PI, 'axis': (1, 0, 0)}, 
    'Y': {'theta': PI, 'axis': (0, 1, 0)}, 
    'Z': {'theta': PI, 'axis': (0, 0, 1)}, 
    'V': {'theta': PI/2, 'axis': (1, 0, 0)}, 
    '-V': {'theta': -PI/2, 'axis': (1, 0, 0)}, 
    'h': {'theta': -PI/2, 'axis': (0, 1, 0)}, 
    '-h': {'theta': PI/2, 'axis': (0, 1, 0)}, 
    'S': {'theta': PI/2, 'axis': (0, 0, 1)}, 
    '-S': {'theta': -PI/2, 'axis': (0, 0, 1)},
    'H_xy': {'theta': PI, 'axis': (1, 1, 0)}, 
    'H_xz': {'theta': PI, 'axis': (1, 0, 1)}, 
    'H_yz': {'theta': PI, 'axis': (0, 1, 1)}, 
    'H_-xy': {'theta': PI, 'axis': (-1, 1, 0)}, 
    'H_x-z': {'theta': PI, 'axis': (1, 0, -1)}, 
    'H_-yz': {'theta': PI, 'axis': (0, -1, 1)},
    'C_xyz': {'theta': 2*PI/3, 'axis': (1, 1, 1)}, 
    '-C_xyz': {'theta': -2*PI/3, 'axis': (1, 1, 1)}, 
    'C_-xyz': {'theta': 2*PI/3, 'axis': (-1, 1, 1)}, 
    '-C_-xyz': {'theta': -2*PI/3, 'axis': (-1, 1, 1)}, 
    'C_x-yz': {'theta': 2*PI/3, 'axis': (1, -1, 1)}, 
    '-C_x-yz': {'theta': -2*PI/3, 'axis': (1, -1, 1)},
    'C_xy-z': {'theta': 2*PI/3, 'axis': (1, 1, -1)}, 
    '-C_xy-z': {'theta': -2*PI/3, 'axis': (1, 1, -1)}
}

# Notes from Zihao(03/30/2023):
# Primitive gate in list should be in time order which will be excuted left to right.
# I generate these decomposition by 'transpile_unitary_to_circuit' and look at the circuit,
# which actually should be better automated.
Clifford_to_primitive = {
    'I': ['I'],
    'X': ['X180_01'],
    'Y': ['Z180_01', 'X180_01'],
    'Z': ['Z180_01'],
    'V': ['X90_01'],
    '-V': ['X-90_01'],
    'h': ['Z90_01', 'X90_01', 'Z270_01'],
    '-h': ['Z270_01', 'X90_01', 'Z90_01'],
    'S': ['Z90_01'],
    '-S': ['Z270_01'],
    'H_xy': ['Z270_01', 'X180_01'],
    'H_xz': ['Z90_01', 'X90_01', 'Z90_01'],
    'H_yz': ['X90_01', 'Z180_01'],
    'H_-xy': ['Z90_01', 'X180_01'],
    'H_x-z': ['Z270_01', 'X90_01', 'Z270_01'],
    'H_-yz': ['Z180_01', 'X90_01'],
    'C_xyz': ['X90_01', 'Z90_01'],
    '-C_xyz': ['Z270_01', 'X-90_01'],
    'C_-xyz': ['Z90_01', 'X-90_01'],
    '-C_-xyz': ['X90_01', 'Z270_01'],
    'C_x-yz': ['Z90_01', 'X90_01'],
    '-C_x-yz': ['X-90_01', 'Z270_01'],
    'C_xy-z': ['Z270_01', 'X90_01'],
    '-C_xy-z': ['X-90_01', 'Z90_01']                
}


#%% Example of generate RB sequeneces

if __name__ == '__main__':
    
    seq_Clifford = generate_RB_Clifford_sequences(Clifford_gates, n_gates=400, n_random=30)
    seq_primitive = generate_RB_primitive_sequences(seq_Clifford, Clifford_to_primitive)
    
    # If you want to check whether the Clifford_sequences is correct.
    seq_index_to_check = 0
    U_list = [Clifford_gates[g]['unitary'] for g in seq_Clifford[seq_index_to_check]]
    result = calculate_combined_unitary(U_list)
    print(result)


#%% Example of decomposing a Clifford gate

if __name__ == '__main__':
    
    gate = 'H_x-z'
    if 'unitary' not in Clifford_gates[gate]: 
        Clifford_gates[gate]['unitary'] = unitary(**Clifford_gates[gate])
    circ = transpile_unitary_to_circuit(Clifford_gates[gate]['unitary'])
    print(circ.draw())
    # See what it print. 
