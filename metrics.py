import numpy as np

def sketch_error(X, X_target):
    return np.linalg.norm(X.T@X-X_target.T@X_target, ord=2)