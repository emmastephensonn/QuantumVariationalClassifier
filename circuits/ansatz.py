import numpy as np


def ansatz(qc, params):
    """
    Apply a parameterized variational ansatz to the quantum circuit.

    The ansatz consists of trainable single-qubit rotations
    followed by entangling CNOT gates.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to which the ansatz is applied.
    params : np.ndarray
        Trainable parameters controlling rotation gates.
    """

    n_qubits = qc.num_qubits

    for i in range(n_qubits):
        qc.ry(params[i], i)

    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)