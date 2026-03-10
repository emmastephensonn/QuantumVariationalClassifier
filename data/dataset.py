import numpy as np
from sklearn .datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_dataset(test_size=0.2):
    """
    Generate and preprocess a binary classification dataset for
    variational quantum experiments.

    Features are scaled to [0, π] for rotation-based encoding,
    labels are mapped from {0,1} to {-1,1}, and the data is split
    into training and test sets.

    Parameters
    ----------
    test_size : float
        Fraction of data used for testing.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Training and test splits of features and labels.
    """
    #Generate synthetic data, add noise so it is not perfectly clean
    X, y = make_moons(n_samples=200, noise=0.1)

    #Scale to 0-pi for rotation angles
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    y = 2* y - 1 #convert {0, 1} -> {-1, 1}

    return train_test_split(X, y, test_size=test_size, random_state=42)

