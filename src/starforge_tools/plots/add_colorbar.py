from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes


def add_colorbar(mappable, label=None, pos="right", size="5%", pad=0.0):
    """From Joseph Long's blog ty https://joseph-long.com/writing/colorbars/"""
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad, axes_class=maxes.Axes)  # note size=X% is different from 0.0X
    cbar = fig.colorbar(mappable, cax=cax, label=label)
    plt.sca(last_axes)
    return cbar
