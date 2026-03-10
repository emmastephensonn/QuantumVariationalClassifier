import numpy as np


def cost(params, X, y, quantum_model):
    """
    Compute the mean squared error loss for the variational quantum classifier.

    For each input sample, the quantum model is evaluated to produce a
    prediction. The cost is calculated as the average squared difference
    between predicted values and the true labels.

    Parameters
    ----------
    params : np.ndarray
        Trainable parameters for the variational quantum circuit.
    X : np.ndarray
        Feature matrix containing input samples.
    y : np.ndarray
        True labels mapped to {-1, 1}.
    quantum_model : callable
        Function that evaluates the quantum circuit and returns
        a prediction for a given input and parameter set.

    Returns
    -------
    float
        Mean squared error between predictions and true labels.
    """
        

    predictions = []

    for x in X:
        pred = quantum_model(x, params)
        predictions.append(pred)

    predictions = np.array(predictions)

    return np.mean((predictions - y) ** 2)