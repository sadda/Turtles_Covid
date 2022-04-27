import sys
sys.path.insert(0, 'src')

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from analysis import plot_confidence, compute_sd, plot_sd, plot_k, compute_ious, plot_reliability0, plot_reliability1, get_entries, prepare_data
from utilities import create_gif
from confidence import confidence_ours
from aggregate import Aggregate


## Load data
data = pandas.read_csv(os.path.join('data', 'aggregated1.csv'))
instagram = pandas.read_csv(os.path.join('data', 'aggregated2.csv'))


## Set parameters
year1 = 2019
year2 = 2020
kwargs = {
    'where_entries': 'boat',
    'where_arrivals': None,
}
ignore_shorter = True
box_shape = True
plot_raw = False
plot_arr = False


## Plot animated confidence intervals
fun = lambda i: plot_confidence(data, year1, year2, i, instagram,
    ignore_shorter = ignore_shorter,
    box_shape = box_shape,
    plot_raw = plot_raw,
    plot_arr = plot_arr,
    return_fig = True,
    **kwargs
)
file_name = os.path.join('figures', 'aggr.gif')
create_gif(fun, range(1,20), file_name)


## Plot animated confidence interval width based on alpha
fun = lambda alpha: plot_sd(data, year1, year2, 30, instagram,
    month1 = 6,
    month2 = 10,    
    rat_max = 10,
    alpha = alpha,
    ymax = 5,
    title = 'Uncertainty parameter alpha = ' + str(alpha),    
    return_fig=True,
    **kwargs
)
file_name = os.path.join('figures', 'alpha.gif')
create_gif(fun, np.round(np.arange(0.01, 0.11, 0.01),2), file_name)


## Plot animated comparison of true and our confidence intervals 1
v1s = np.power(10, np.linspace(1,4,100))
fun = lambda rat: plot_k(v1s, v1s*rat, 1e-4, 1e-4,
    ymin = 0,
    ymax = 2,
    xscale='log',
    title='Ratio of DETECTED viewing pressures v2/v1 = ' + str(rat),
    return_fig=True
)
file_name = os.path.join('figures', 'confidence1.gif')
create_gif(fun, np.round(np.arange(0.1, 1.1, 0.1),1), file_name)


## Plot animated comparison of true and our confidence intervals 2
v1s = np.power(10, np.linspace(0,3,100))
fun = lambda v2: plot_k(v1s, v2*np.ones(v1s.shape), 1e-4, 1e-4,
    ymin = 0,
    ymax = 2,
    xscale='log',
    title='DETECTED viewing pressure v2 = ' + str(v2),
    return_fig=True
)
file_name = os.path.join('figures', 'confidence2.gif')
create_gif(fun, np.arange(1, 11, 1), file_name)


## Plot the reliability graph (decrease n to make it master)
v1s = np.arange(1, 51, 1)
v2s = np.arange(1, 51, 1)
m = compute_ious(v1s, v2s, 1e-4, 1e-4, n=1000000)
plot_reliability0(m, v1s, v2s, title='IoU of confidence intervals', xlabel='$\mathrm{v}_{\mathrm{detected}}^{\mathrm{year}_1}$', ylabel='$\mathrm{v}_{\mathrm{detected}}^{\mathrm{year}_2}$')
tikzplotlib.save(os.path.join('figures', 'reliability1.tex'))
plt.savefig(os.path.join('figures', 'reliability1.png'), bbox_inches='tight')
plot_reliability1(m, v1s, v2s)
plt.savefig(os.path.join('figures', 'reliability2.png'), bbox_inches='tight')


## Save to csv reliability data
dict = {'v1': v1s, 'IoU': [np.min(m[i:,i:]) for i in range(len(v1s))]}
df = pandas.DataFrame(dict)
df.to_csv(os.path.join('figures', 'reliability2.csv'), index=False)


##
month1 = 6
month2 = 9
x_max = 200
cs = ['yellow', 'blue', 'black']

v1s = np.arange(1, 201, 1)
v2s = np.arange(1, 201, 1)

p1 = p2 = 1e-4
m2 = np.zeros((len(v1s), len(v2s)))
for (i, v1) in enumerate(v1s):
    for (j, v2) in enumerate(v2s):
        _, lb1, ub1 = confidence_ours(v1, v2, p1, p2)
        m2[i,j] = np.sqrt(ub1 / lb1)

for (year1, year2) in zip((2019, 2019), (2020, 2021)):    
    plot_reliability0(m2, v1s, v2s,
        xlabel='$\mathrm{v}_{\mathrm{detected}}^{%d}$' % year1,
        ylabel='$\mathrm{v}_{\mathrm{detected}}^{%d}$' % year2,
        title='confidence interval width',
        boundaries=np.linspace(1,3,11)
    )

    plt.plot([0, x_max], [5, 5], c='black', linestyle='dashed')
    plt.plot([5, 5], [0, x_max], c='black', linestyle='dashed')

    for i, n_aggr in enumerate([1, 7, 15]):
        entries, arrivals, years, months = get_entries(data, **kwargs)

        x1 = entries[(years == year1) * (months >= month1) * (months <= month2)]
        x2 = entries[(years == year2) * (months >= month1) * (months <= month2)]

        aggregate = Aggregate(month1, month2, n_aggr, **kwargs)
        x1 = aggregate.split(x1)
        x2 = aggregate.split(x2)
        if n_aggr == 1:
            label_name = '1 day'
        else:
            label_name = str(n_aggr) + ' days'
        plt.scatter(x1, x2, c=cs[i], label=label_name)

    plt.xlim([0, x_max])
    plt.ylim([0, x_max])
    plt.legend()
    tikzplotlib.save(os.path.join('figures', 'width_%d_%d.tex' % (year1, year2)))
    plt.savefig(os.path.join('figures', 'width_%d_%d.png' % (year1, year2)), bbox_inches='tight')


## Save to csv confidence intervals width 
for (year1, year2) in zip((2019, 2019), (2020, 2021)):
    conf_size = compute_sd(data, year1, year2, 30, instagram,
        month1=6,
        month2=10,
        alpha=0.05,
        **kwargs
    )
    conf_size = np.array(conf_size).T

    dict = {'aggrv': range(1,len(conf_size[0])+1), 'm6': conf_size[0], 'm7': conf_size[1], 'm8': conf_size[2], 'm9': conf_size[3], 'm10': conf_size[4]}
    df = pandas.DataFrame(dict)
    df.to_csv(os.path.join('figures', 'confidence_%s_%s.csv' % (year1, year2)), index=False)


## Save to csv confidence intervals
ignore_shorter=True
for (year1, year2) in zip((2019, 2019), (2020, 2021)):
    for n_aggr in (1, 7, 15, 184):
        for box_shape in (True, False):
            entries, arrivals, years, months = get_entries(data, **kwargs)
            c0, c1, c2, t = prepare_data(entries, years, year1, year2, months, min(months), max(months), n_aggr, instagram, ignore_shorter=ignore_shorter, box_shape=box_shape)
            c0_no_inst, _, _, _ = prepare_data(entries, years, year1, year2, months, min(months), max(months), n_aggr, None, ignore_shorter=ignore_shorter, box_shape=box_shape)
            c3, _, _, _ = prepare_data(arrivals, years, year1, year2, months, min(months), max(months), n_aggr, None, ignore_shorter=ignore_shorter, box_shape=box_shape)

            df = pandas.DataFrame({'x': t, 'photos_lo': c1, 'photos_up': c2, 'photos_point': c0,
                'arrivals_exact': c3,
                'photos1': np.full(len(t), np.nan),
                'photos2': np.full(len(t), np.nan),
                'photos_point_initial': c0_no_inst})

            file_name = "Ratio_%d_%d_%s_%02d_%s.csv" % (year2,year1,kwargs['where_entries'],n_aggr,box_shape)
            df.to_csv(file_name, index=None)

