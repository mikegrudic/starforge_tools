"""Implements a matplotlib plot consisting of multiple panels, with a timelapse from left to right and
stacking the different maps atop one another.
"""

from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy.spatial import KDTree
import numpy as np
from .map_renderer import MapRenderer
from .add_colorbar import add_colorbar
from .star_markers import plot_star_markers, plot_star_legend
from .io import get_pdata_for_maps, get_snapshot_timeline
from . import rendermaps


def multipanel_timelapse_map(
    output_dir=".",
    maps=rendermaps.DEFAULT_MAPS,
    times=4,
    res=1024,
    box_frac=0.24,
    colorbar_frac=0.05,
    figsize=(8, 8),
    plot_stars=True,
    SFE_in_title=True,
    relative_times=True,
    cmap_limits={},
):
    """Make a multi-panel map where rows plot a certain kind of coloarmap plot with colorbar, and
    each column is a certain simulation time.

    Parameters
    ----------
    output_dir: string, optional
        Directory path where the simulation snapshots are
    maps: iterable, optional
        Iterable of which maps to plot in each row. Defaults to plotting surface density, velocity dispersion,
        temperature, and magnetic energy fraction.

        Each entry must be an object defined in starforge_tools.plots.rendermaps with attributes plotlabel,
        required_datafields, colormap, render, and cmap_default_limits. See e.g. starforge_tools.plots.rendermaps.SurfaceDensity
    times: optional
        Either an integer number of evenly-spaced simulation times for each column, or an array_like of simulation times in
        whatever units the snapshots are in, or as an astropy quantity. If the exact simulation times are not available,
        selects the closest possible time. (default: 4)
    res: int, optional
        Resolution of the maps (default: 1024)
    box_frac: float or array_like, optional
        Fraction of the total box size that the maps will plot
    colorbar_frac: float, optional
        Width of the colorbar relative to panel size (default: 0.1)
    figsize: tuple, optional
        Size of figure in inches (default: (8,8))
    plot_stars: boolean, optional
        Whether to plot star markers (default: True)
    SFE_in_title: boolean, optional
        Whether to plot SFE in the column titles in addition to the time (default: True)
    relative_times: boolean, optional
        Whether specified times are relative to that of the first snapshot (default: True)
    limits: dict, optional
        Dictionary wholes keys are the names of rendermaps and whose entries are shape (2,) tuples specifying the
        upper and lower limits of the colormap, overriding the default.
    """
    snappaths, snaptimes = get_snapshot_timeline(output_dir)
    if relative_times:
        snaptimes -= snaptimes[0]

    if isinstance(times, int):  # if we specified an integer number of times,
        times = np.linspace(snaptimes.min(), snaptimes.max(), times)  # snaptimes[:: len(snaptimes) // (times - 1)]
    _, ngb_idx = KDTree(np.c_[snaptimes]).query(np.c_[times])
    times = snaptimes[ngb_idx]
    snaps = snappaths[ngb_idx]

    num_maps, num_times = len(maps), len(times)

    if isinstance(box_frac, float):
        box_frac = np.repeat(box_frac, num_times)

    # re-scaling the first N-1 panels so that they are all the
    # same size when the colorbar rescales the panel hosting the bar
    # I will burn for eternity for this.
    width_ratios = (num_times - 1) * [1 / (1 + colorbar_frac)] + [1]  # EVIL GROSS DON'T DO THIS
    fig, ax = plt.subplots(num_maps, num_times, figsize=figsize, gridspec_kw={"width_ratios": width_ratios})

    if num_times == 1:
        ax = np.atleast_2d(ax).T
    elif num_maps == 1:
        ax = np.atleast_2d(ax)

    for i, t in enumerate(times):
        pdata = get_pdata_for_maps(snaps[i], maps)
        boxsize = pdata["Header"]["BoxSize"]
        length = boxsize * box_frac[i]
        mapargs = {"size": length, "res": res, "center": np.zeros(3)}
        X, Y = 2 * [np.linspace(-0.5 * length, 0.5 * length, res)]
        X, Y = np.meshgrid(X, Y, indexing="ij")
        if "PartType5/Masses" in pdata:
            SFE = pdata["PartType5/Masses"].sum() / (0.8 * pdata["PartType0/Masses"].sum())
        else:
            SFE = 0

        title = t.to_string(formatter="%4.3g", format="latex")
        if SFE_in_title:
            title += rf", SFE=${round(SFE*100,0)}\%$"
        ax[0, i].set_title(title)

        renderer = MapRenderer(pdata, mapargs)
        for j, mapname in enumerate(maps):
            axes = ax[j, i]
            axes.set_aspect("equal")
            render, default_limits, cmap, label = renderer.get_render_items(mapname)
            if mapname in cmap_limits:
                limits = cmap_limits[mapname]
            else:
                limits = default_limits

            if limits[0]:
                if limits[0] < 0 or np.log10(limits[1] / limits[0]) < 1:
                    norm = None
                else:
                    norm = colors.LogNorm(*limits)
                    vmin = vmax = None
            else:
                norm = None
                vmin, vmax = limits
            pcm = axes.pcolormesh(X, Y, render, norm=norm, cmap=cmap, label=label, vmin=vmin, vmax=vmax)
            if i == num_times - 1:
                add_colorbar(pcm, label=label, size=str(100 * colorbar_frac) + "%")
            if i == 0:
                axes.set(ylabel=r"$z\rm  \,\left(pc\right)$")
                if plot_stars and j == 0:
                    plot_star_legend(axes)
            if j == num_maps - 1:
                axes.set(xlabel=r"$x\rm  \,\left(pc\right)$")  # ,ylabel=r'$\rm Y \,\left(pc\right)$')
            if i > 0:  # and j < num_maps - 1:
                axes.set_yticklabels([])
            if j < num_maps - 1:
                axes.set_xticklabels([])

            if "PartType5/Masses" in pdata and plot_stars:
                plot_star_markers(axes, pdata)

            axes.set(xlim=[-0.5 * length, 0.5 * length], ylim=[-0.5 * length, 0.5 * length])

    # fig.tight_layout(h_pad=0, w_pad=0)
    fig.subplots_adjust(hspace=-0.0, wspace=0.0)
    plt.savefig("multipanel.png")  # , bbox_inches="tight")
