import numpy as np
from scipy.stats import nbinom, norm


def confidence_true(v1, v2, p1, p2, alpha=0.05, n=10000):
    # This generate number of failures
    u1 = nbinom.rvs(v1, p1, size=n)
    u2 = nbinom.rvs(v2, p2, size=n)
    # We need to add number of successes
    rat = np.sort((u2+v2) / (u1+v2))

    return np.mean(rat), rat[int(n*alpha/2)], rat[int(n*(1-alpha/2))]


def confidence_ours(v1, v2, p1, p2, alpha=0.05):
    c0, c1, c2 = compute_confidence([v1], [v2], alpha=alpha, p_ratio=p2/p1)

    return c0[0], c1[0], c2[0]


def compute_confidence(x1, x2, alpha=0.05, c_max=10, p_ratio=1, **kwargs):
    k = norm.ppf(1-alpha/2)
    c0 = np.zeros(len(x1))
    c1 = np.zeros(len(x1))
    c2 = np.zeros(len(x1))
    for (i, (x1_i, x2_i)) in enumerate(zip(x1, x2)):
        if x1_i == 0 or x2_i == 0:
            if x1_i == 0 and x2_i == 0:
                c0_i = np.nan
            elif x1_i == 0:
                c0_i = c_max
            else:
                c0_i = 0
            c1_i = 0
            c2_i = c_max
        else:
            c0_i = x2_i / x1_i / p_ratio
            c1_i = x2_i / x1_i / p_ratio * np.exp(-k * np.sqrt(1/x1_i + 1/x2_i))
            c2_i = x2_i / x1_i / p_ratio * np.exp(k * np.sqrt(1/x1_i + 1/x2_i))
        c0[i] = c0_i
        c1[i] = c1_i
        c2[i] = c2_i
    return c0, c1, c2

