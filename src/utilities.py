import numpy as np
import ipywidgets as widgets
import imageio


def generate_slider(mode='float', **kwargs):
    if mode == 'float':
        return widgets.FloatSlider(**kwargs)
    elif mode == 'int':
        return widgets.IntSlider(**kwargs)
    else:
        raise(Exception('Slider mode should be float or int'))   


def generate_slider_int(**kwargs):
    return widgets.FloatSlider(**kwargs)


def fig_to_matrix(fig):
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def create_gif(fun, par_range, file_name, fps=3):
    imageio.mimsave(file_name, [fig_to_matrix(fun(i)) for i in par_range], fps=fps)

