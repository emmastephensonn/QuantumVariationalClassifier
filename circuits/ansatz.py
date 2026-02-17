import numpy as np
from qiskit import QuantumCircuit

def variational_layer(params):
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

    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0,1)

    return qc