import sys
sys.path.insert(0, 'src')

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from analysis import plot_confidence, plot_sd, plot_k, compute_ious, plot_reliability0, plot_reliability1
from utilities import create_gif

data = pandas.read_csv(os.path.join('data', 'aggregated1.csv'))
instagram = pandas.read_csv(os.path.join('data', 'aggregated2.csv'))

year1 = 2019
year2 = 2020
place = None
ignore_shorter = True
box_shape = True
plot_raw = False
plot_arr = False

fun = lambda i: plot_confidence(data, year1, year2, i, instagram, place=place, return_fig=True, ignore_shorter=ignore_shorter, box_shape=box_shape, plot_raw=plot_raw, plot_arr=plot_arr)
file_name = os.path.join('figures', 'aggr.gif')
create_gif(fun, range(1,20), file_name)



n_aggr_max = 30

fun = lambda alpha: plot_sd(data, year1, year2, n_aggr_max, instagram, month1=6, rat_max=10, alpha=alpha, ymax=5, title='Uncertainty parameter alpha = ' + str(alpha), return_fig=True)
file_name = os.path.join('figures', 'alpha.gif')
create_gif(fun, np.round(np.arange(0.01, 0.11, 0.01),2), file_name)



v1s = np.power(10, np.linspace(1,4,100))

fun = lambda rat: plot_k(v1s, v1s*rat, 1e-4, 1e-4, ymin=0, ymax = 2, xscale='log', title='Ratio of DETECTED viewing pressures v2/v1 = ' + str(rat), return_fig=True)
file_name = os.path.join('figures', 'confidence1.gif')
create_gif(fun, np.round(np.arange(0.1, 1.1, 0.1),1), file_name)



v1s = np.power(10, np.linspace(0,3,100))

fun = lambda v2: plot_k(v1s, v2*np.ones(v1s.shape), 1e-4, 1e-4, ymin=0, ymax = 2, xscale='log', title='DETECTED viewing pressure v2 = ' + str(v2), return_fig=True)
file_name = os.path.join('figures', 'confidence2.gif')
create_gif(fun, np.arange(1, 11, 1), file_name)



v1s = np.arange(1, 51, 1)
v2s = np.arange(1, 51, 1)
m = compute_ious(v1s, v2s, 1e-4, 1e-4, n=1000000)
plot_reliability0(m, v1s, v2s, title='IoU of confidence intervals')
plt.savefig(os.path.join('figures', 'reliability1.png'), bbox_inches='tight')
plot_reliability1(m, v1s, v2s)
plt.savefig(os.path.join('figures', 'reliability2.png'), bbox_inches='tight')



'''
from analysis import get_entries, prepare_data

ignore_shorter=True

for (year1, year2) in zip((2019, 2019), (2020, 2021)):
    for place in ('boat', 'underwater'):
        for n_aggr in (1, 7, 15, 184):
            for box_shape in (True, False):
                entries, arrivals, years, months = get_entries(data, place=place)
                c0, c1, c2, t = prepare_data(entries, years, year1, year2, months, min(months), max(months), n_aggr, instagram, ignore_shorter=ignore_shorter, box_shape=box_shape)
                c0_no_inst, _, _, _ = prepare_data(entries, years, year1, year2, months, min(months), max(months), n_aggr, None, ignore_shorter=ignore_shorter, box_shape=box_shape)
                c3, _, _, _ = prepare_data(arrivals, years, year1, year2, months, min(months), max(months), n_aggr, None, ignore_shorter=ignore_shorter, box_shape=box_shape)

                df = pandas.DataFrame({'x': t, 'photos_lo': c1, 'photos_up': c2, 'photos_point': c0,
                    'arrivals_exact': c3,
                    'photos1': np.full(len(t), np.nan),
                    'photos2': np.full(len(t), np.nan),
                    'photos_point_initial': c0_no_inst})

                file_name = "Ratio_%d_%d_%s_%02d_%s.csv" % (year2,year1,place,n_aggr,box_shape)
                df.to_csv(file_name, index=None)
'''
