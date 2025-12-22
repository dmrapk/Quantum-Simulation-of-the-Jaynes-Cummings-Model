
from qiskit.quantum_info import Operator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from scipy.linalg import expm

def get_binary_hamiltonian_rabi(N_cutoff, g, wc, wa, model='rabi',omega_c=1.0, omega_a=1.0):
    dim = N_cutoff
    mat_a_dag = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        mat_a_dag[n+1, n] = np.sqrt(n + 1)
    
    op_a_dag_cav = SparsePauliOp.from_operator(Operator(mat_a_dag))
    op_a_cav = op_a_dag_cav.adjoint()
    
    op_I_atom = SparsePauliOp("I")
    op_Z_atom = SparsePauliOp("Z")
    op_X_atom = SparsePauliOp("X")
    op_Y_atom = SparsePauliOp("Y")
    
    op_sm_atom = 0.5 * (op_X_atom + 1j * op_Y_atom) 
    op_sp_atom = op_sm_atom.adjoint()
    
    num_op = op_a_dag_cav.compose(op_a_cav)
    H_cav = omega_c * op_I_atom.tensor(num_op)
    
    op_I_cav = SparsePauliOp("I" * op_a_dag_cav.num_qubits)
    H_atom = (omega_a / 2.0) * (op_I_atom - op_Z_atom).tensor(op_I_cav)
    
    int_1 = op_sm_atom.tensor(op_a_dag_cav) 
    int_2 = op_sp_atom.tensor(op_a_cav)     
    
    if model == 'jc':
        H_int = g * (int_1 + int_2)
    else: 
        int_cr_1 = op_sp_atom.tensor(op_a_dag_cav) 
        int_cr_2 = op_sm_atom.tensor(op_a_cav)
        H_int = g * (int_1 + int_2 + int_cr_1 + int_cr_2)
        
    return (H_cav + H_atom + H_int).simplify()


def get_exact_dynamics(H_mat, times, num_qubits, psi0):
    n_vals = []
    pe_vals = []
    atom_index = num_qubits - 1 
    cavity_mask = (1 << atom_index) - 1 
    
    for t in times:
        psi_t = expm(-1j * H_mat * t) @ psi0
        probs = np.abs(psi_t)**2
        
        n_avg = 0
        pe_avg = 0
        for k, p in enumerate(probs):
            if p == 0: continue
            if (k >> atom_index) & 1: pe_avg += p
            n_avg += (k & cavity_mask) * p
            
        n_vals.append(n_avg)
        pe_vals.append(pe_avg)
    return np.array(n_vals), np.array(pe_vals)
