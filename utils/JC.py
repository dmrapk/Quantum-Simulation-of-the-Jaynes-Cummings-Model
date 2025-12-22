import numpy as np
from numpy import sqrt, exp
from scipy.linalg import expm
from scipy.special import factorial
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from utils.PauliGadget import trotter_1order

def H_eff_matrix(n, omega_c, omega_a, g):
    """Return the 2x2 effective Hamiltonian for JC sector with total excitations n (n>=1)."""
    # Basis: [ |e, n-1>, |g, n> ]
    E1 = omega_c * (n - 1) + 0.5 * omega_a  # energy of |e, n-1>
    E2 = omega_c * n - 0.5 * omega_a  # energy of |g, n>
    off = g * sqrt(n)
    H = np.array([[E1, off],
                  [off, E2]], dtype=complex)
    return H

def pauli_coeffs_from_2x2(H):
    """
    For a 2x2 Hermitian H, return coefficients (c0, cx, cz) so that
    H = c0 * I + cx * X + cz * Z  
    """
    a = H[0,0].real
    b = H[1,1].real
    off = H[0,1].real
    c0 = 0.5 * (a + b)
    cz = 0.5 * (a - b)
    cx = off
    return c0, cx, cz


def get_pauli_op(H):
    a, b = H[0,0].real, H[1,1].real
    off = H[0,1].real
    return SparsePauliOp.from_list([('I', 0.5*(a+b)), ('X', off), ('Z', 0.5*(a-b))])

def get_exact_solution(H, times):
    psi0 = np.array([1, 0], dtype=complex)
    pe_exact = []
    for t in times:
        pe_exact.append(np.abs((expm(-1j * H * t) @ psi0)[0])**2)
    return np.array(pe_exact)

def get_Pe_coherent(times, mu, cutoff):
    Pe_t = np.zeros_like(times)
    
    weights = []
    for n in range(cutoff):
        prob = exp(-mu) * (mu**n) / factorial(n)
        weights.append(prob)
        
    weights = np.array(weights)
    weights /= np.sum(weights) 

    for i, t in enumerate(times):
        sum_val = 0.0
        for n in range(cutoff):
            Omega = 2 * 0.1 * sqrt(n + 1)
            sum_val += weights[n] * np.cos(Omega * t)
        
        Pe_t[i] = 0.5 * (1.0 + sum_val)
        
    return Pe_t

def get_Pe_coherent_exact(times, mu, cutoff=60):
    Pe_t = np.zeros_like(times)
    weights = np.array([np.exp(-mu)*(mu**n)/factorial(n) for n in range(cutoff)])
    weights /= weights.sum()
    
    for i, t in enumerate(times):
        # Sum over Rabi oscillations cos(2*g*sqrt(n+1)*t)
        osc = 0.0
        for n in range(cutoff):
            omega_n = 2 * 0.1 * np.sqrt(n + 1)
            osc += weights[n] * np.cos(omega_n * t)
        Pe_t[i] = 0.5 * (1.0 + osc)
    return Pe_t

def get_binary_jc_hamiltonian(N_cutoff, g, omega_c, omega_a):
    """
    Constructs the JC Hamiltonian for Binary Encoding with cutoff N_cutoff.
    Register layout: [q_atom, q_cav_highest, ..., q_cav_lowest]
    """

    dim = N_cutoff
    a_dag_matrix = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        a_dag_matrix[n+1, n] = np.sqrt(n + 1)
        
    op_a_dag = SparsePauliOp.from_operator(Operator(a_dag_matrix))
    op_a = op_a_dag.adjoint()
    
    num_cav_qubits = int(np.ceil(np.log2(dim)))
    I_cav = SparsePauliOp("I" * num_cav_qubits)
    
    Z_atom = SparsePauliOp("Z")
    X_atom = SparsePauliOp("X")
    Y_atom = SparsePauliOp("Y")
    
    sig_p = 0.5 * (X_atom + 1j * Y_atom)
    sig_m = 0.5 * (X_atom - 1j * Y_atom)
 
    num_op = op_a_dag.compose(op_a)
    H_cav = omega_c * num_op.tensor(SparsePauliOp("I")) 
    
    H_at = (omega_a / 2.0) * I_cav.tensor(Z_atom)
    

    int_1 = op_a_dag.tensor(sig_m)
    int_2 = op_a.tensor(sig_p)
    H_int = g * (int_1 + int_2)
    
    H_total = H_cav + H_at + H_int
    return H_total.simplify()

def get_initial_state(alpha, N_cutoff):

    coeffs = []
    norm = 0
    for n in range(N_cutoff):
        c = np.exp(-abs(alpha)**2/2) * (alpha**n) / np.sqrt(factorial(n))
        coeffs.append(c)
        norm += abs(c)**2
    
    coeffs = np.array(coeffs) / np.sqrt(norm)
    

    atom_state = np.array([0, 1])

    full_state = np.kron(atom_state, coeffs)
    return full_state