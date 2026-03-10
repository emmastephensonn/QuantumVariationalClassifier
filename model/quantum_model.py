import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from circuits.encode import encode_data
from circuits.ansatz import ansatz

simulator = AerSimulator()


def quantum_model(x, params):
    """
    Evaluate the variational quantum classifier for a single input.

    A quantum circuit is constructed using feature encoding
    followed by a parameterized ansatz. Measurement outcomes
    are used to estimate the expectation value of Z.

    Parameters
    ----------
    x : np.ndarray
        Input feature vector.
    params : np.ndarray
        Trainable parameters for the ansatz.

    Returns
    -------
    float
        Prediction value in the range [-1, 1].
    """

    n_qubits = len(x)

    qc = QuantumCircuit(n_qubits)

    encode_data(qc, x)
    ansatz(qc, params)

    qc.measure_all()

    job = simulator.run(qc, shots=1024)
    result = job.result()

    counts = result.get_counts()

    expectation = 0

    for bitstring, count in counts.items():
        value = 1 if bitstring[-1] == "0" else -1
        expectation += value * count

    expectation /= 1024

    return expectation