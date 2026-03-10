from qiskit import QuantumCircuit


def encode_data(qc, x):
    """
    Encode classical features into a quantum circuit using
    rotation-based angle encoding.

    Each feature is mapped to a qubit rotation around the Y axis.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit used for encoding.
    x : np.ndarray
        Feature vector scaled to [0, π].
    """

    for i, value in enumerate(x):
        qc.ry(value, i)