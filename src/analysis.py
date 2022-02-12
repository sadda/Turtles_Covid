import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from aggregate import Aggregate
from confidence import confidence_ours, confidence_true, compute_confidence


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


def plot_k(v1s, v2s, p1, p2, **kwargs):
    m1 = np.zeros(len(v1s))
    m2 = np.zeros(len(v1s))
    lb1 = np.zeros(len(v1s))
    lb2 = np.zeros(len(v1s))
    ub1 = np.zeros(len(v1s))
    ub2 = np.zeros(len(v1s))
    
    for (i, (v1, v2)) in enumerate(zip(v1s, v2s)):        
        m1[i], lb1[i], ub1[i] = confidence_ours(v1, v2, p1, p2)
        m2[i], lb2[i], ub2[i] = confidence_true(v1, v2, p1, p2)

    return plot_k0(m1, m2, v1s, lb1, lb2, ub1, ub2, **kwargs)


def plot_k0(m1, m2, v1s, lb1, lb2, ub1, ub2, ymin=None, ymax=None, xscale=None, return_fig=False, title=None, **kwargs):
    fig = plt.figure(facecolor=(1, 1, 1))
    plot_between(v1s, lb1, ub1, color='blue', alpha=0.2)
    plot_between(v1s, lb2, ub2, color='red', alpha=0.2)
    plt.plot(v1s, m1, color='black')
    plt.plot(v1s, m2, color='black', linestyle='dotted')    

    plt.legend(('Point estimate ours', 'Point estimate true', 'Confidence interval ours', 'Confidence interval true'))
    plt.xlabel('Detected viewing pressure')
    plt.ylabel('Ratio of real viewing pressures')
    plt.title(title)

    if xscale == 'log':
        plt.xscale("log")
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if ymax is not None:
        plt.ylim(top=ymax)    

    if return_fig:        
        return fig


def plot_between(x, y1, y2, **kwargs):
    plt.plot(x, y1, label='_nolegend_', **kwargs)
    plt.plot(x, y2, label='_nolegend_', **kwargs)
    plt.fill_between(x, y1, y2, **kwargs)
