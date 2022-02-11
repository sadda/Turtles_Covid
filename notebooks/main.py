import sys
sys.path.insert(0, 'src')

import os
import pandas
from analysis import plot_confidence
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
