# Quantum Simulation of Light-Matter Interaction Models

**Hamiltonian simulation of bosonic systems** — final project for the course **"Quantum Optics"** (Theoretical Physics, M.Sc.), University of Porto.

---

## About

This repository contains the final maunscript alonside relevant code respective to the final project for the "Quantum Optics" course at the University of Porto.

It establishes a rigorous framework for simulating infinite-dimensional bosonic light-matter interaction systems on finite-dimensional digital quantum hardware. Addressing the challenge of the infinite bosonic Hilbert space, implementing and comparing various truncation and qubit encoding strategies, including unary, binary, and symmetry-aware reductions. To ensure simulation fidelity, the framework incorporates rigorous error analysis techniques, utilizing tail-control guarantees for initial state preparation and dynamical leakage bounds to manage wavefunction spreading during time evolution. The Hamiltonian dynamics are simulated via Trotterization, employing explicit decompositions of truncated bosonic operators into Pauli strings to balance resource efficiency with algorithmic precision.These methods are applied to the Jaynes-Cummings and Quantum Rabi models to reproduce distinct non-classical phenomena. The simulations successfully capture the characteristic $\sqrt{n}$ scaling of Rabi frequencies and the collapse and revival of Rabi oscillations in coherent states, demonstrating the quantization of the electromagnetic field. Furthermore, the project extends to the ultrastrong coupling regime of the Quantum Rabi model, where the breakdown of the Rotating Wave Approximation leads to dynamical leakage. In this regime, the simulation validates the theoretical prediction of vacuum instability, observing the generation of excitations from the ground state due to counter-rotating terms.

---

## Repository contents

- `main.ipynb` — Jupyter notebook containing the simulations and analysis.  
- `main.pdf` — "article" manuscript.  
- `figures/` — generated figures used.  
- `utils/` — helper modules and utility scripts used by the notebook.  

---

## Requirements

This project uses standard scientific Python packages. Notably

- Jupyter / JupyterLab  
- `numpy`, `scipy`, `matplotlib`  
- `Qiskit` (version used: 2.1.1)
- `Qiskit_Aer` (version used: 0.17.1)

