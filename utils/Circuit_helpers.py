from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from utils.PauliGadget import trotter_1order
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def get_noise_model(t1, t2, gate_time=0.1):
    noise_model = NoiseModel()
    error_1q = thermal_relaxation_error(t1, t2, gate_time)
    noise_model.add_all_qubit_quantum_error(error_1q, ['id', 'rz', 'sx', 'x', 'h'])
    error_cx = thermal_relaxation_error(t1, t2, gate_time * 2)
    error_2q = error_cx.tensor(error_cx)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model


def run_trotter_sim(H_op, times,num_qubits=7, shots=2048):
    backend = AerSimulator()
    n_sim = []
    pe_sim = []
    for t in times:
        qc = QuantumCircuit(num_qubits)
        target_dt = 0.003 
        steps = max(1, int(t / target_dt))
        trotter_1order(qc, range(num_qubits), H_op, t, trotter_steps=steps)
        qc.measure_all()

        transpiled = transpile(qc, backend=backend, optimization_level=1)
        result = backend.run(transpiled, shots=shots).result()
        counts = result.get_counts()

        avg_n = 0
        avg_pe = 0
        for bitstring, count in counts.items():
            if bitstring[0] == '1':
                avg_pe += count
            avg_n += int(bitstring[1:], 2) * count
        n_sim.append(avg_n / shots)
        pe_sim.append(avg_pe / shots)

    return n_sim, pe_sim
