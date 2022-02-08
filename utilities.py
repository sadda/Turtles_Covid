import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets


def generate_data():
    np.random.seed(666)
    n = 184
    t = np.linspace(0, np.pi, n)
    x1 = 0.5*np.sin(t) + 0.5*np.random.rand(n)
    x2 = np.sin(t) + 0.5*np.random.rand(n)
    
    return t, x1, x2


def generate_slider():
    return widgets.IntSlider(value=5, min=1, max=20, step=1)


def aggregate(x, step):
    t = np.arange(0, len(x), step)
    x_aggr = np.array([np.sum(x[t[i]:t[i+1]]) for i in range(len(t)-1)])
    t_aggr = t[:-1] + (t[1] - t[0])/2

    return x_aggr, t_aggr


def plot(x1, x2, n_aggr, return_fig=True):
    x1_aggr, t_aggr = aggregate(x1, n_aggr)
    x2_aggr, _ = aggregate(x2, n_aggr)

    k = 1
    c0 = x1_aggr / x2_aggr
    c1 = x1_aggr / x2_aggr * np.exp(k * (1/x1_aggr + 1/x2_aggr))
    c2 = x1_aggr / x2_aggr * np.exp(-k * (1/x1_aggr + 1/x2_aggr))

    fig = plt.figure(facecolor=(1, 1, 1))
    ax = fig.add_subplot(111)
    plt.plot(t_aggr, c0)
    plt.plot(t_aggr, c1, color="grey", alpha=0.3)
    plt.plot(t_aggr, c2, color="grey", alpha=0.3)
    plt.fill_between(t_aggr, c1, c2, color="grey", alpha=0.3)

    plt.xlim(1, len(x1))    
    plt.ylim(0, 3)
    plt.title('Aggregation days = ' + str(n_aggr))

    months = [1, 32, 62, 93, 124, 154]
    ax.set_xticks(months)
    ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90) 

    dx = 0.4; dy = 0.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    if return_fig:
        return fig


def fig_to_matrix(fig):
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image



