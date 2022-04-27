import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from aggregate import Aggregate
from confidence import confidence_ours, confidence_true, compute_confidence
from utilities import IoU


def prepare_data(entries, years, year1, year2, months, month1, month2, n_aggr, instagram=None, **kwargs):
    x1 = entries[(years == year1) * (months >= month1) * (months <= month2)]
    x2 = entries[(years == year2) * (months >= month1) * (months <= month2)]

    aggregate = Aggregate(month1, month2, n_aggr, **kwargs)
    x1 = aggregate.split(x1)
    x2 = aggregate.split(x2)
    if not instagram is None:
        p1 = instagram.users.values[instagram.year == year1][0]
        p2 = instagram.users.values[instagram.year == year2][0]
        c0, c1, c2 = compute_confidence(x1, x2, p_ratio=p2/p1, **kwargs)
    else:
        c0, c1, c2 = compute_confidence(x1, x2, p_ratio=1, **kwargs)

    t, (c0, c1, c2) = aggregate.convert_to_plot((c0, c1, c2), **kwargs)
    
    return c0, c1, c2, t


def get_entries(data, where_entries=None, where_arrivals=None, **kwargs):
    if where_entries is None:
        entries = data.total.values
    elif where_entries == "boat":
        entries = data.boat.values
    elif where_entries == "underwater":
        entries = data.underwater.values
    else:
        raise(Exception('where_entries should be None, boat or underwater'))
    if where_arrivals is None:
        arrivals = data.arr_total.values
    elif where_arrivals == "airport":
        arrivals = data.arr_airport.values
    elif where_arrivals == "port":
        arrivals = data.arr_port.values
    else:
        raise(Exception('where_arrivals should be None, airport or port'))
    years = data.year.values
    months = data.month.values
    return entries, arrivals, years, months


def plot_confidence(data, year1, year2, n_aggr, instagram, plot_raw=False, plot_arr=False, **kwargs):
    entries, arrivals, years, months = get_entries(data, **kwargs)
    c0, c1, c2, t = prepare_data(entries, years, year1, year2, months, min(months), max(months), n_aggr, instagram, **kwargs)
    if plot_raw and instagram is not None:
        c0_raw, _, _, _ = prepare_data(entries, years, year1, year2, months, min(months), max(months), n_aggr, None, **kwargs)
    else:
        c0_raw = None
    if plot_arr:
        kwargs2 = copy.deepcopy(kwargs)
        kwargs2.update({'box_shape': False})
        c3, _, _, t3 = prepare_data(arrivals, years, year1, year2, months, min(months), max(months), 1, None, **kwargs2)
    else:
        c3, t3 = None, None

    return_fig = kwargs.pop('return_fig', False)
    title = kwargs.pop('title', 'Aggregation days = ' + str(n_aggr))
    xmax = kwargs.pop('xmax', len(entries[years==year1]))
    ymax = kwargs.pop('ymax', 3)
    xticks_offset = kwargs.pop('xticks_offset', 0.4)
    return plot_confidence0(t, c0, c1, c2, c0_raw, t3, c3,
        return_fig=return_fig,
        title=title,
        xmax=xmax,
        ymax=ymax,
        xticks_offset=xticks_offset,
        **kwargs,
        )


def plot_confidence0(t, c0, c1, c2, c0_raw=None, t3=None, c3=None, return_fig=True, title=None, xmax=None, ymax=None, xticks_offset=0, **kwargs):
    fig = plt.figure(facecolor=(1, 1, 1))
    ax = fig.add_subplot(111)
    plt.plot(t, c0, label='Ratio entries after Instagram')
    plot_between(t, c1, c2, color="grey", alpha=0.2)
    if c0_raw is not None:
        plt.plot(t, c0_raw, label='Ratio enties raw')
    if c3 is not None:
        plt.plot(t3, c3, color='black', linestyle='dashed', label='Ratio arrivals')
    
    if xmax is None:
        xmax = len(t)        
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.xlim(1, xmax)
    plt.title(title)
    plt.legend(loc='upper right')

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


def compute_sd(data, year1, year2, n_aggr_max, instagram, month1=5, month2=10, rat_max=10, **kwargs):
    entries, _, years, months = get_entries(data, **kwargs)
            
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

    return res

def plot_sd(data, year1, year2, n_aggr_max, instagram, month1=5, month2=10, **kwargs):
    res = compute_sd(data, year1, year2, n_aggr_max, instagram, month1=month1, month2=month2, **kwargs)

    legend = kwargs.pop('legend', Aggregate.month_names[range(month1-1, month2)])
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
    plt.xlabel('DETECTED viewing pressure v1')
    plt.ylabel('Ratio of REAL viewing pressures v2/v1')
    plt.title(title)

    if xscale == 'log':
        plt.xscale("log")
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if ymax is not None:
        plt.ylim(top=ymax)    

    if return_fig:        
        return fig


def compute_ious(v1s, v2s, p1, p2, n=100000, **kwargs):
    m = np.zeros((len(v1s), len(v2s)))
    for (i, v1) in enumerate(v1s):
        for (j, v2) in enumerate(v2s):
            _, lb1, ub1 = confidence_ours(v1, v2, p1, p2)
            _, lb2, ub2 = confidence_true(v1, v2, p1, p2, n=n)
            m[i,j] = IoU(lb1, ub1, lb2, ub2)
    return m


def plot_reliability(v1s, v2s, p1, p2, **kwargs):
    m = compute_ious(v1s, v2s, p1, p2, **kwargs)
    plot_reliability0(m, v1s, v2s, **kwargs)


def plot_reliability0(m, v1s, v2s, xlabel='v1', ylabel='v2', return_fig=False, title=None, **kwargs):
    fig = plt.figure(facecolor=(1, 1, 1))
    plt.imshow(np.flipud(m), cmap=plt.cm.RdBu, extent=(min(v1s)-1, max(v1s), min(v2s)-1, max(v2s)))
    plt.colorbar(boundaries=np.linspace(0,1,11)) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if return_fig:        
        return fig


def plot_reliability1(m, v1s, v2s, return_fig=False, title=None, **kwargs):
    fig = plt.figure(facecolor=(1, 1, 1))
    if v1s.shape != v2s.shape or not all(np.equal(v1s, v2s)):
        raise Exception('v1s and v2s must ocnincide')
    data = [np.min(m[i:,i:]) for i in range(len(v1s))]
    plt.plot(v1s, data)
    plt.xlabel('v_detected')
    plt.ylabel('IoU')
    plt.ylim(0, 1)
    plt.xlim(0, max(v1s))
    plt.title(title)    

    if return_fig:        
        return fig


def plot_between(x, y1, y2, **kwargs):
    plt.plot(x, y1, label='_nolegend_', **kwargs)
    plt.plot(x, y2, label='_nolegend_', **kwargs)
    plt.fill_between(x, y1, y2, **kwargs)
