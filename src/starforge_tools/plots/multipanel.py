"""Implements a matplotlib plot consisting of multiple panels, with a timelapse from left to right and
stacking the different maps atop one another.
"""

import os
import hashlib
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy.spatial import KDTree
import numpy as np
from .map_renderer import MapRenderer
from .add_colorbar import add_colorbar
from .star_markers import plot_star_markers, plot_star_legend
from .io import get_pdata_for_maps, get_snapshot_timeline
from . import rendermaps


def _apply_los_slice(pdata: dict, slice_thickness: float) -> dict:
    """Restrict per-particle arrays to |z_los| < slice_thickness/2 along the projection axis (pdata index 2)."""
    out = dict(pdata)
    for i in range(6):
        coord_key = f"PartType{i}/Coordinates"
        if coord_key not in pdata:
            continue
        coords = pdata[coord_key]
        mask = np.abs(coords[:, 2]) < 0.5 * slice_thickness
        prefix = f"PartType{i}/"
        for k, v in pdata.items():
            if k.startswith(prefix) and hasattr(v, "__len__") and len(v) == len(coords):
                out[k] = v[mask]
    return out


def _cache_path(cache_dir, snappath, mapname, res, supersample, box_frac, slice_thickness):
    """Build the cache filename for a single post-downsample rendered map. Path-hash prefix lets snapshots
    with identical basenames in different output dirs coexist without collision."""
    path_hash = hashlib.md5(str(snappath).encode()).hexdigest()[:8]
    snapname = os.path.basename(snappath).replace(".hdf5", "")
    slice_str = f"{slice_thickness:.4f}" if slice_thickness is not None else "full"
    fname = f"{path_hash}_{snapname}_{mapname}_res{res}_ss{supersample}_bf{box_frac:.4f}_slice{slice_str}.npy"
    return os.path.join(cache_dir, fname)


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
    slice_thickness=None,
    supersample=1,
    cache_dir=".maps_cache",
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
    slice_thickness: float, optional
        If set, restrict all particle data to |line-of-sight coord| < slice_thickness/2 before rendering, so that
        each panel shows a thin slab through the projection axis (pdata index 2) instead of the full box.
        Units are the snapshot's native length (pc for STARFORGE). Applies to gas and stars alike. (default: None)
    supersample: int, optional
        Render each map at supersample*res internally, then block-average down to res for display.
        Provides anti-aliasing at the cost of supersample**2 more compute and memory per panel. (default: 1)
    cache_dir: str or None, optional
        Directory for caching the post-downsample (res, res) rendered map arrays as .npy files. Cache keys
        include the snapshot path, map name, res, supersample, box_frac, and slice_thickness, so any of those
        changing produces a fresh render. Entries older than the snapshot's mtime are ignored.
        Pass None to disable caching. (default: ".maps_cache")
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
        if slice_thickness is not None:
            pdata = _apply_los_slice(pdata, slice_thickness)
        boxsize = pdata["Header"]["BoxSize"]
        length = boxsize * box_frac[i]
        mapargs = {"size": length, "res": res * supersample, "center": np.zeros(3)}
        X, Y = 2 * [np.linspace(-0.5 * length, 0.5 * length, res)]
        X, Y = np.meshgrid(X, Y, indexing="ij")
        if "PartType5/Masses" in pdata:
            SFE = pdata["PartType5/Masses"].sum() / (0.8 * pdata["PartType0/Masses"].sum() + pdata["PartType5/Masses"].sum())
        else:
            SFE = 0

        title = t.to_string(formatter="%4.3g", format="latex")
        if SFE_in_title:
            title += rf", SFE=${round(SFE * 100, 0)}\%$"
        ax[0, i].set_title(title)

        cached_renders = {}
        cache_paths = {}
        if cache_dir is not None:
            snap_mtime = os.path.getmtime(snaps[i])
            for mapname in maps:
                cp = _cache_path(cache_dir, snaps[i], mapname, res, supersample, box_frac[i], slice_thickness)
                cache_paths[mapname] = cp
                if os.path.isfile(cp) and os.path.getmtime(cp) > snap_mtime:
                    raw = np.load(cp).astype(np.float64)
                    cached_renders[mapname] = 10.0**raw if getattr(rendermaps, mapname).log_cache else raw
        missing = [m for m in maps if m not in cached_renders]
        renderer = MapRenderer(pdata, mapargs) if missing else None

        for j, mapname in enumerate(maps):
            axes = ax[j, i]
            axes.set_aspect("equal")
            if mapname in cached_renders:
                render = cached_renders[mapname]
                cls = getattr(rendermaps, mapname)
                default_limits, cmap, label = cls.cmap_default_limits, cls.colormap, cls.plotlabel
            else:
                assert renderer is not None  # guaranteed: missing is non-empty iff we entered this branch
                render, default_limits, cmap, label = renderer.get_render_items(mapname)
                if supersample > 1:
                    render = render.reshape(res, supersample, res, supersample).mean(axis=(1, 3))
                if cache_dir is not None:
                    os.makedirs(cache_dir, exist_ok=True)
                    if getattr(rendermaps, mapname).log_cache:
                        with np.errstate(divide="ignore"):
                            np.save(cache_paths[mapname], np.log10(render).astype(np.float16))
                    else:
                        np.save(cache_paths[mapname], render)
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
            pcm = axes.pcolormesh(X, Y, render, norm=norm, cmap=cmap, label=label, vmin=vmin, vmax=vmax, rasterized=True)
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
    # rasterized pcolormesh in a vector PDF: pick dpi so each panel keeps its full res pixels.
    panel_inches = min(figsize[0] / num_times, figsize[1] / num_maps)
    dpi = int(np.ceil(res / panel_inches))
    plt.savefig("multipanel.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
