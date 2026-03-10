import numpy as np
from sklearn.metrics import accuracy_score


def evaluate(X_test, y_test, params, quantum_model):
    """
    Evaluate the trained variational quantum classifier.

    Predictions are generated for the test set and
    classification accuracy is computed.

    Parameters
    ----------
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test labels.
    params : np.ndarray
        Optimized circuit parameters.
    quantum_model : callable
        Quantum model used for predictions.

    Returns
    -------
    float
        Classification accuracy.
    """

    predictions = []

    for x in X_test:
        pred = quantum_model(x, params)
        predictions.append(1 if pred >= 0 else -1)

    return accuracy_score(y_test, predictions)