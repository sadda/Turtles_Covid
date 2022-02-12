import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from aggregate import Aggregate


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


def prepare_data(entries, years, year1, year2, months, month1, month2, n_aggr, instagram=None, **kwargs):
    x1 = entries[(years == year1) * (months >= month1) * (months <= month2)]
    x2 = entries[(years == year2) * (months >= month1) * (months <= month2)]

    aggregate = Aggregate(month1, month2, n_aggr, **kwargs)
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
    months = data.month.values
    return entries, years, months


def plot_confidence(data, year1, year2, n_aggr, instagram, **kwargs):
    entries, years, months = get_entries(data, **kwargs)
    c0, c1, c2, t = prepare_data(entries, years, year1, year2, months, min(months), max(months), n_aggr, instagram, **kwargs)

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
    plot_between(t, c1, c2, color="grey", alpha=0.2)

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



def plot_sd(data, year1, year2, n_aggr_max, instagram, month1=5, month2=10, rat_max=10, **kwargs):
    entries, years, months = get_entries(data, **kwargs)
            
    res = []
    for n_aggr in range(1, n_aggr_max+1):
        _, c1, c2, _ = prepare_data(entries, years, year1, year2, months, month1, month2, n_aggr, instagram, ignore_month=False, **kwargs)
        aggregate = Aggregate(month1, month2, n_aggr, ignore_month=False, **kwargs)

        res_month = []
        for month in range(month1, month2+1):
            with np.errstate(divide='ignore'):
                qwe = np.sqrt(c2 / c1)
            asd = qwe[aggregate.months == month]
            asd[asd == np.inf] = rat_max
            res_month.append(np.mean(asd))
        res.append(res_month)

    legend = kwargs.pop('legend', Aggregate.month_names[range(month1, month2+1)])
    return plot_sd0(range(1, n_aggr_max+1), np.array(res), legend=legend, **kwargs)


def plot_sd0(t, res, ymax=None, legend=None, title=None, return_fig=False, **kwargs):
    fig = plt.figure(facecolor=(1, 1, 1))
    plt.plot(t, res)
    xmin = np.min(t)-1
    xmax = np.max(t)+1
    plt.hlines(1, xmin, xmax, linestyles='dotted')
    plt.xlim(xmin, xmax)
    if ymax is not None:
        plt.ylim(0.9, ymax)
    else:
        plt.ylim(bottom=0.9)
    
    plt.xlabel('Aggregation days')
    plt.ylabel('Confidence interval size')
    plt.legend(legend)
    plt.title(title)
    
    if return_fig:        
        return fig


def plot_between(x, y1, y2, **kwargs):
    plt.plot(x, y1, label='_nolegend_', **kwargs)
    plt.plot(x, y2, label='_nolegend_', **kwargs)
    plt.fill_between(x, y1, y2, **kwargs)