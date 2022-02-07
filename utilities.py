import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


def generate_data():
    np.random.seed(666)
    n = 184
    t = np.linspace(0, np.pi, n)
    x1 = 0.5*np.sin(t) + 0.5*np.random.rand(n)
    x2 = np.sin(t) + 0.5*np.random.rand(n)
    slider = widgets.IntSlider(value=5, min=1, max=10, step=1)

    return t, x1, x2, slider


def aggregate(x, step):
    t = np.arange(0, len(x), step)
    x_aggr = np.array([np.sum(x[t[i]:t[i+1]]) for i in range(len(t)-1)])
    t_aggr = t[:-1] + (t[1] - t[0])/2

    return x_aggr, t_aggr


def plot(x1, x2, n_aggr):
    x1_aggr, t_aggr = aggregate(x1, n_aggr)
    x2_aggr, _ = aggregate(x2, n_aggr)

    k = 1
    c0 = x1_aggr / x2_aggr
    c1 = x1_aggr / x2_aggr * np.exp(k * (1/x1_aggr + 1/x2_aggr))
    c2 = x1_aggr / x2_aggr * np.exp(-k * (1/x1_aggr + 1/x2_aggr))

    plt.plot(t_aggr, c0)
    plt.plot(t_aggr, c1, color="grey", alpha=0.3)
    plt.plot(t_aggr, c2, color="grey", alpha=0.3)
    plt.fill_between(t_aggr, c1, c2, color="grey", alpha=0.3)

    plt.xlim(1, len(x1))    
    plt.ylim(0, 3)

