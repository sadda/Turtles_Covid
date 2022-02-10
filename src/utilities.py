import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from aggregate import *


def generate_slider():
    return widgets.IntSlider(value=5, min=1, max=20, step=1)


def compute_confidence(x1, x2, k=1.96, c_max=10, p_ratio=1, **kwargs):
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


def prepare_data(entries, years, year1, year2, n_aggr, instagram=None, **kwargs):
    x1 = entries[years == year1]
    x2 = entries[years == year2]

    aggregate = Aggregate(5, 10, n_aggr, **kwargs)
    x1 = aggregate.split(x1)
    x2 = aggregate.split(x2)

    if not instagram is None:
        p1 = instagram.users.values[instagram.year == year1][0]
        p2 = instagram.users.values[instagram.year == year1][0]
        c0, c1, c2 = compute_confidence(x1, x2, p_ratio=p1/p2, **kwargs)
    else:
        c0, c1, c2 = compute_confidence(x1, x2, p_ratio=1, **kwargs)

    t, (c0, c1, c2) = aggregate.convert_to_plot((c0, c1, c2), **kwargs)

    return c0, c1, c2, t


def get_entries(data, place=None, **kwargs):
    if place is None:
        entries = data.total.values
    elif place == "boat":
        entries = data.boat.values
    elif place == "underwater":
        entries = data.underwater.values
    else:
        raise(Exception('place should be None, boat or underwater'))
    years = data.year.values
    return entries, years


def plot_confidence(data, year1, year2, n_aggr, instagram, **kwargs):
    entries, years = get_entries(data, **kwargs)
    c0, c1, c2, t = prepare_data(entries, years, year1, year2, n_aggr, instagram, **kwargs)

    return_fig = kwargs.pop('return_fig', False)
    title = kwargs.pop('title', 'Aggregation days = ' + str(n_aggr))
    xmax = kwargs.pop('xmax', len(entries[years==year1]))
    ymax = kwargs.pop('ymax', 3)
    xticks_offset = kwargs.pop('xticks_offset', 0.4)
    return plot_confidence0(c0, c1, c2, t,
        return_fig=return_fig,
        title=title,
        xmax=xmax,
        ymax=ymax,
        xticks_offset=xticks_offset,
        **kwargs,
        )


def plot_confidence0(c0, c1, c2, t, return_fig=True, title=None, xmax=None, ymax=None, xticks_offset=0, **kwargs):
    fig = plt.figure(facecolor=(1, 1, 1))
    ax = fig.add_subplot(111)
    plt.plot(t, c0)
    plt.plot(t, c1, color="grey", alpha=0.3)
    plt.plot(t, c2, color="grey", alpha=0.3)
    plt.fill_between(t, c1, c2, color="grey", alpha=0.3)    

    if xmax is None:
        xmax = len(t)        
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.xlim(1, xmax)
    plt.title(title)

    plt.hlines(1, 1, xmax, linestyles='dotted')
    
    months = [1, 32, 62, 93, 124, 154]
    ax.set_xticks(months)
    ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90) 

    if xticks_offset != 0:
        offset = matplotlib.transforms.ScaledTranslation(xticks_offset, 00, fig.dpi_scale_trans)
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
    
    if return_fig:
        return fig


def fig_to_matrix(fig):
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image



