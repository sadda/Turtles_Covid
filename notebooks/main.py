import sys
sys.path.insert(0, 'src')

import os
import pandas
import numpy as np
from analysis import plot_confidence, plot_sd, plot_k
from utilities import create_gif

data = pandas.read_csv(os.path.join('data', 'aggregated1.csv'))
instagram = pandas.read_csv(os.path.join('data', 'aggregated2.csv'))

year1 = 2019
year2 = 2020
place = None
ignore_shorter = True
box_shape = True

fun = lambda i: plot_confidence(data, year1, year2, i, instagram, place=place, return_fig=True, ignore_shorter=ignore_shorter, box_shape=box_shape)
file_name = os.path.join('figures', 'aggr.gif')
create_gif(fun, range(1,20), file_name)



n_aggr_max = 30

fun = lambda alpha: plot_sd(data, year1, year2, n_aggr_max, instagram, month1=6, rat_max=10, alpha=alpha, ymax=5, title='Uncertainty parameter alpha = ' + str(alpha), return_fig=True)
file_name = os.path.join('figures', 'alpha.gif')
create_gif(fun, np.round(np.arange(0.01, 0.11, 0.01),2), file_name)



v1s = np.power(10, np.linspace(1,4,100))

fun = lambda rat: plot_k(v1s, v1s*rat, 1e-4, 1e-4, ymin=0, ymax = 2, xscale='log', title='Ratio of detected viewing pressures = ' + str(rat), return_fig=True)
file_name = os.path.join('figures', 'confidence.gif')
create_gif(fun, np.round(np.arange(0.1, 1.1, 0.1),1), file_name)
