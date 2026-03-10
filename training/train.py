import numpy as np
from scipy.optimize import minimize
from training.cost import cost


def train(X_train, y_train, quantum_model, n_params, maxiter=100):
    """
    Train the variational quantum classifier by optimizing circuit parameters.

    A classical optimizer minimizes the cost function by iteratively updating
    the parameters of the variational quantum circuit.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels mapped to {-1, 1}.
    quantum_model : callable
        Function that evaluates the quantum circuit and returns
        a prediction for a given input and parameter set.
    n_params : int
        Number of trainable parameters in the ansatz.
    maxiter : int
        Maximum number of optimization iterations.

    Returns
    -------
    params_opt : np.ndarray
        Optimized circuit parameters.
    loss_history : list
        Cost value at each optimization step.
    """

    # initialize parameters randomly
    init_params = np.random.uniform(0, 2*np.pi, n_params)

    loss_history = []

    def objective(params):
        loss = cost(params, X_train, y_train, quantum_model)
        loss_history.append(loss)
        return loss

    result = minimize(
        objective,
        init_params,
        method="COBYLA",
        options={"maxiter": maxiter}
    )

    params_opt = result.x

    return params_opt, loss_history