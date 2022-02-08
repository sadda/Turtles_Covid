import os
import imageio
from utilities import *

t, x1, x2 = generate_data()

fps = 3
kwargs_write = {'fps':fps, 'quantizer':'nq'}
imageio.mimsave(os.path.join('results', 'aggr.gif'), [fig_to_matrix(plot(x1, x2, i)) for i in range(1,20)], fps=fps)
