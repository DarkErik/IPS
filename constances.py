import numpy as np

AGE_CATEGORIES = [10, 18, 30, 45, 65, 999]
AGE_RESULTING_LABELS = [
    np.array([1, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0, 1]),
]