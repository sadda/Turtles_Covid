import numpy as np
from scipy.stats import nbinom
from analysis import compute_confidence

def generate1(v1, v2, p1, p2, alpha=0.05, n=10000):
    u1 = nbinom.rvs(v1, p1, size=n)
    u2 = nbinom.rvs(v2, p2, size=n)
    rat = np.sort(u2 / u1)

    return np.mean(rat), rat[int(n*alpha/2)], rat[int(n*(1-alpha/2))]


def generate2(v1, v2, p1, p2, alpha=0.05, n=10000):
    c0, c1, c2 = compute_confidence([v1], [v2], alpha=alpha, p_ratio=1)

    return c0[0], c1[0], c2[0]
