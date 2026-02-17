from qiskit import QuantumCircuit

def encode_data(x):
    """
    Encode a 2D classical input vector into a quantum circuit
    using rotation-based feature mapping.

    Each feature is mapped to an Ry rotation on a separate qubit.

    Parameters
    ----------
    x : array-like of length 2
        Input feature vector scaled to [0, π].

    Returns
    -------
    QuantumCircuit
        A 2-qubit circuit with data encoded via Ry rotations
    """

    qc = QuantumCircuit(2)

    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    return qc