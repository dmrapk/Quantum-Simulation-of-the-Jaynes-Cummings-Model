import numpy as np
from numpy import sqrt

def fock_annihilation_matrix(N_dim):
    A = np.zeros((N_dim, N_dim), dtype=complex)
    for n in range(1, N_dim):
        A[n-1, n] = sqrt(n)
    return A

def unary_basis_index(n, N_dim):
    """Return computational-basis integer index for unary |n> in N_dim qubits.
       We use bitstring where bit n is '1' and others '0', with bitstring left-to-right
       corresponding to qubit ordering used by our Operator construction.
    """
    bits = ['0'] * N_dim
    bits[n] = '1'
    bitstr = ''.join(bits)
    return int(bitstr, 2)

def build_unary_operator_from_small_matrix(mat):
    """Embed an N x N operator (Fock basis) into the 2^N unary qubit space."""
    N_dim = mat.shape[0]
    D = 2 ** N_dim
    M = np.zeros((D, D), dtype=complex)
    for i in range(N_dim):
        ui = unary_basis_index(i, N_dim)
        for j in range(N_dim):
            uj = unary_basis_index(j, N_dim)
            M[ui, uj] = mat[i, j]
    return M