"""Univariate DTW and timestamp-aware TWED, compiled with numba.

TWED: Marteau, TPAMI 2009, doi:10.1109/TPAMI.2008.76.
The virtual sample preceding each series has value/time zero.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def dtw(x, tx, y, ty, nu=0.1, penalty=0.01):
    """sqrt(sum squared local costs), full unconstrained alignment."""
    n, m = len(x), len(y)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return np.sqrt(previous[m])


@njit(cache=True)
def twed(x, tx, y, ty, nu=0.1, penalty=0.01):
    """p=1 TWED; timestamps in seconds; nu > 0 and penalty >= 0."""
    n, m = len(x), len(y)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        xp = 0.0 if i == 1 else x[i - 2]
        tp = 0.0 if i == 1 else tx[i - 2]
        delete_x = abs(x[i - 1] - xp) + nu * (tx[i - 1] - tp) + penalty
        for j in range(1, m + 1):
            yp = 0.0 if j == 1 else y[j - 2]
            sp = 0.0 if j == 1 else ty[j - 2]
            delete_y = abs(y[j - 1] - yp) + nu * (ty[j - 1] - sp) + penalty
            match = (abs(x[i - 1] - y[j - 1]) + abs(xp - yp)
                     + nu * (abs(tx[i - 1] - ty[j - 1]) + abs(tp - sp)))
            current[j] = min(previous[j] + delete_x, current[j - 1] + delete_y,
                             previous[j - 1] + match)
        previous = current
    return previous[m]


@njit(cache=True)
def pairwise(values, timestamps, lengths, use_twed=False, nu=0.1, penalty=0.01):
    n = len(lengths)
    output = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            x, y = values[i, :lengths[i]], values[j, :lengths[j]]
            tx, ty = timestamps[i, :lengths[i]], timestamps[j, :lengths[j]]
            if use_twed:
                distance = twed(x, tx, y, ty, nu, penalty)
            else:
                distance = dtw(x, tx, y, ty)
            output[i, j] = output[j, i] = distance
    return output
