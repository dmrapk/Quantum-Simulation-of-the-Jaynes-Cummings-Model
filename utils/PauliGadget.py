import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

def trotter_1order(qc, target_register, sparse_pauli_op, time, trotter_steps):
    dt = time / trotter_steps
    pauli_list = sparse_pauli_op.paulis
    coeffs = sparse_pauli_op.coeffs
    for _ in range(trotter_steps):
        for pauli_obj, coeff in zip(pauli_list, coeffs):
            coeff = np.real(coeff)
            label = pauli_obj.to_label()
            if set(label) == {"I"}:
                qc.global_phase += -coeff * dt
                continue
            involved = []
            for i, char in enumerate(reversed(label)):
                if char != 'I': involved.append((i, target_register[i], char))
            for _, q, char in involved:
                if char == 'X': qc.h(q)
                elif char == 'Y': qc.sdg(q); qc.h(q)
            parity_qubit = involved[-1][1]
            for _, q, _ in involved[:-1]: qc.cx(q, parity_qubit)
            
            qc.rz(2.0 * coeff * dt, parity_qubit)
            for _, q, _ in reversed(involved[:-1]): qc.cx(q, parity_qubit)
            for _, q, char in reversed(involved):
                if char == 'X': qc.h(q)
                elif char == 'Y': qc.h(q); qc.s(q)