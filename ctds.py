"""Serial CTDS-style embedding, nearest-center Voronoi implementation.

Based on https://github.com/LeisureGong/CTDS (PAKDD 2024).
This pilot uses two 64-point regions, each with 32-point sliding windows,
at a common 50 Hz grid: region 1.28 s, pattern window 0.64 s.
No radius rejection, no learned shapelets, no full NxN kernel matrix.
"""
import random
import numpy as np
from scipy.spatial.distance import cdist
from numpy.lib.stride_tricks import sliding_window_view


def embedding(series, seed, psi=128, repetitions=50):
    if series.ndim != 2 or series.shape[1] != 128:
        raise ValueError('This pilot expects N x 128 univariate sequences.')
    rng = random.Random(seed)
    features = []
    for start in [0, 64]:
        windows = sliding_window_view(series[:, start:start + 64], 32, axis=1).copy()
        # Match sklearn.preprocessing.scale(axis=1) on each window.
        from sklearn.preprocessing import scale
        n, q, d = windows.shape
        points = scale(windows.reshape(n * q, d), axis=1)
        if len(points) < psi:
            raise ValueError('Not enough windows for psi.')
        histograms = np.zeros((n, repetitions * psi))
        for repeat in range(repetitions):
            centers = points[rng.sample(range(len(points)), psi)]
            ids = cdist(points, centers).argmin(axis=1).reshape(n, q)
            for i in range(n):
                histograms[i, repeat * psi:(repeat + 1) * psi] = np.bincount(ids[i], minlength=psi) / q
        norms = np.linalg.norm(histograms, axis=1, keepdims=True)
        features.append(histograms / np.maximum(norms, 1e-15))
    return np.concatenate(features, axis=1)
