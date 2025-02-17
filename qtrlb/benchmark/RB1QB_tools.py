import secrets
import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator


PI = np.pi
CLIFFORD_SET_1QB = {
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
# I generate these decomposition by 'transpile_unitary_to_circuit' and actually looking at the circuit.
# The primitive gate set here is ['X180', 'X90', 'X-90', 'Z180', 'Z90', 'Z270', 'I'].
# In principle, each Clifford gate should take finite time.
# By our convention, Identity take finite time and upd_param, where Z gate take no time in RB.
# So 'Z', 'S', '-S' should be decomposed with addition of Identity gate.
CLIFFORD_TO_PRIMITIVE = {
    'I': ['I'],
    'X': ['X180'],
    'Y': ['Z180', 'X180'],
    'Z': ['Z180', 'I'],
    'V': ['X90'],
    '-V': ['X-90'],
    'h': ['Z90', 'X90', 'Z270'],
    '-h': ['Z270', 'X90', 'Z90'],
    'S': ['Z90', 'I'],
    '-S': ['Z270', 'I'],
    'H_xy': ['Z270', 'X180'],
    'H_xz': ['Z90', 'X90', 'Z90'],
    'H_yz': ['X90', 'Z180'],
    'H_-xy': ['Z90', 'X180'],
    'H_x-z': ['Z270', 'X90', 'Z270'],
    'H_-yz': ['Z180', 'X90'],
    'C_xyz': ['X90', 'Z90'],
    '-C_xyz': ['Z270', 'X-90'],
    'C_-xyz': ['Z90', 'X-90'],
    '-C_-xyz': ['X90', 'Z270'],
    'C_x-yz': ['Z90', 'X90'],
    '-C_x-yz': ['X-90', 'Z270'],
    'C_xy-z': ['Z270', 'X90'],
    '-C_xy-z': ['X-90', 'Z90']                
}




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
                                 basis_gates: tuple = ('x', 'z', 'rx', 'rz')) -> QuantumCircuit:
    """
    Using Qiskit transpile function to decompose a unitary into primitive gate.
    It may fail if the primitive gate set is not universal. 
    Reference:
    https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html
    https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html
    """
    circ = QuantumCircuit(1)
    circ.append(Operator(U), [0])
    circ = transpile(circ, basis_gates=list(basis_gates), optimization_level=3)
    return circ
    
    
def calculate_combined_unitary(U_list: list[np.ndarray] | np.ndarray,
                               eliminate_float_error: bool = True) -> np.ndarray:
    """
    Expect to get a list/ndarray of unitary (ndarray) in time order (left one happen first).
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


def find_Clifford_gate(U: np.ndarray, Clifford_set: dict) -> str:
    """
    Given an expression of unitary, find the name of its corresponding Clifford gate.
    Allow a difference with global phase.
    """
    for k, v in Clifford_set.items():
        V = v['unitary']
        idx = np.unravel_index(np.argmax(np.abs(U), axis=None), U.shape)  # Index of maximum element
        if not np.allclose(np.abs(U[idx]), np.abs(V[idx])): continue
        global_phase = V[idx] / U[idx]
        if np.allclose(U * global_phase, V): return k
    
    raise ValueError('There is no such Clifford gate!')
    
    
def generate_RB_Clifford_gates(n_gates: int, Clifford_set: dict = CLIFFORD_SET_1QB) -> list[str]:
    """
    Generate a list of n_gates Clifford gates. Each entry is a string of Clifford gate names..
    Notice the dictionary Clifford_sets will be changed in situ (calculate its unitary).
    """
    # Calculate unitary if it's not in Clifford_set.
    for v in Clifford_set.values():
        if 'unitary' in v: continue
        v['unitary'] = unitary(v['theta'], v['axis'])
        
    if n_gates == 0: return []
        
    Clifford_gates = []
    Clifford_gates_mat = np.zeros(shape=(n_gates, 2, 2), dtype='complex128')
    
    # Random choice from Clifford_set
    for i in range(n_gates):
        key_random = secrets.choice(list(Clifford_set.keys()))
        Clifford_gates.append(key_random)
        Clifford_gates_mat[i] = Clifford_set[key_random]['unitary']
        
    # Find the inverse gate and add it to the end of gate list.
    total_U = calculate_combined_unitary(Clifford_gates_mat)
    inverse = find_Clifford_gate(total_U.T.conj(), Clifford_set)
    Clifford_gates.append(inverse)
        
    return Clifford_gates


def generate_RB_primitive_gates(Clifford_gates: list[str],
                                remove_identity: bool = False,
                                Clifford_to_primitive: dict = CLIFFORD_TO_PRIMITIVE) -> list[str]:
    """
    Transpile all Clifford gate in a gate list into primitive_gate.
    Return list with uncertain length.
    """
    # Iterable unpacking cannot be used in comprehension.
    primitive_gates = []
    for Clifford_gate in Clifford_gates:
        primitive_gates.extend(Clifford_to_primitive[Clifford_gate])
        
    return optimize_circuit(primitive_gates, remove_identity)
    

def optimize_circuit(gates: list[str], remove_identity: bool = False) -> list[str]:
    """
    Combine all adjacent Z gates, and if required, remove all Identity.
    I'm sorry. This is for Qblox. Hopefully it's fast enough.
    """
    if remove_identity: gates = [gate for gate in gates if gate != 'I']
    optimized_gates = []
    
    i = 0
    while i < len(gates):
        # If it's X or I gate, when we just keep it
        if gates[i].startswith(('X', 'I')): 
            optimized_gates.append(gates[i])
            i += 1
            
        # If it's Z gate, we want to know how much consecutive Z we have here.
        elif gates[i].startswith('Z'):
            
            # This j tells number of consecutive Z gates.
            for j, gate in enumerate(gates[i:]):
                if gate.startswith(('X', 'I')): break
            
            # If we only find one Z gate, keep it
            if j <= 1: 
                optimized_gates.append(gates[i])
                i += 1

            # If there is more than one, we calculate angle and keep only one gate.
            else:
                angle = np.sum([int(gate[1:]) for gate in gates[i:i+j]]) % 360
                optimized_gates.append(f'Z{angle}') 
                i += j
        
        # Protection
        else:
            raise ValueError(f'Cannot process/optimize gate {gates[i]}!')
            
    if remove_identity: optimized_gates = [gate for gate in optimized_gates if gate != 'I']
    return optimized_gates




####################################################################################################
# Example of generate RB gates
if __name__ == '__main__':

    Clifford_gates = generate_RB_Clifford_gates(n_gates=400)
    primitive_gates = generate_RB_primitive_gates(Clifford_gates)
    
    # If you want to check whether the Clifford_gates is correct.
    seq_index_to_check = 0
    U_list = [CLIFFORD_SET_1QB[gate]['unitary'] for gate in Clifford_gates[seq_index_to_check]]
    result = calculate_combined_unitary(U_list)
    print(result)


# Example of decomposing a single Clifford gate
if __name__ == '__main__':

    gate = 'H_x-z'
    if 'unitary' not in CLIFFORD_SET_1QB[gate]: 
        CLIFFORD_SET_1QB[gate]['unitary'] = unitary(**CLIFFORD_SET_1QB[gate])
    circ = transpile_unitary_to_circuit(CLIFFORD_SET_1QB[gate]['unitary'])
    print(circ.draw())
    # See what it print. 

