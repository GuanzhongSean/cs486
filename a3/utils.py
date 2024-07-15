import numpy as np

def normalize(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized
