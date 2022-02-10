import sys
sys.path.insert(0, 'src')

import os
import pandas
import imageio
from utilities import *

data = pandas.read_csv(os.path.join('data', 'aggregated1.csv'))
instagram = pandas.read_csv(os.path.join('data', 'aggregated2.csv'))

year1 = 2019
year2 = 2020
place = None
ignore_shorter = True
box_shape = True

fps = 3
kwargs_write = {'fps':fps, 'quantizer':'nq'}
imageio.mimsave(os.path.join('figures', 'aggr.gif'), [fig_to_matrix(plot_confidence(data, year1, year2, i, instagram, place=place, return_fig=True, ignore_shorter=ignore_shorter, box_shape=box_shape)) for i in range(1,20)], fps=fps)
