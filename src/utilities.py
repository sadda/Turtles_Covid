import numpy as np
import ipywidgets as widgets
import imageio


def generate_slider():
    return widgets.IntSlider(value=5, min=1, max=20, step=1)


def fig_to_matrix(fig):
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def create_gif(fun, par_range, file_name, fps=3):
    imageio.mimsave(file_name, [fig_to_matrix(fun(i)) for i in par_range], fps=fps)

