"""Implements a matplotlib plot consisting of multiple panels, with a timelapse from left to right and
stacking the different maps atop one another.
"""

from glob import glob
from os.path import isfile
from natsort import natsorted
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import h5py
from astropy import units as u
from scipy.spatial import KDTree
import numpy as np
from .map_renderer import MapRenderer
from .add_colorbar import add_colorbar
from .star_markers import plot_star_markers, plot_star_legend
from . import rendermaps

DEFAULT_MAPS = (
    "SurfaceDensity",
    "VelocityDispersion",
    "MassWeightedTemperature",
    "AlfvenSpeed",
    #    "XCoordinate",
    #  "ZCoordinate",
)


def get_pdata_for_maps(snapshot_path: str, maps=DEFAULT_MAPS, centering=None) -> dict:
    """Does the I/O to get the data required for the specified maps"""
    required_data = set.union(*[getattr(rendermaps, s).required_datafields for s in maps])
    snapdata = {}
    with h5py.File(snapshot_path, "r") as F:
        snapdata["Header"] = dict(F["Header"].attrs)
        for i in range(6):
            s = f"PartType{i}"
            if s not in F.keys():
                continue
            for k in F[s].keys():
                s2 = s + "/" + k
                if s2 in required_data or i == 5:  # always get star data because it doesn't take much space
                    snapdata[s2] = F[s2][:]

                    # transformation so we are projecting to the x-z plane
                    data = snapdata[s2]
                    if len(data.shape) == 2 and data.shape[-1] == 3:
                        snapdata[s2] = np.c_[data[:, 0], data[:, 2], data[:, 1]]
                    if "Coordinates" in s2:
                        snapdata[s2] -= snapdata["Header"]["BoxSize"] * 0.5
                        if centering is not None:
                            

    return snapdata


def get_snapshot_units(F):
    """Given an h5py file instance for a snapshot, returns a dictionary
    whose entries are astropy quantities giving the unit length, speed, mass, and magnetic field for the snapshot.

    Parameters
    ----------
    F: h5py.File
        h5py file instance for the snapshot
    Returns:
        Dictionary with keys "Length", "Speed", "Mass", and "MagneticField" giving the unit quantities for the simulation.
    """
    unit_length = F["Header"].attrs["UnitLength_In_CGS"] * u.cm
    unit_speed = F["Header"].attrs["UnitVelocity_In_CGS"] * u.cm / u.s
    unit_mass = F["Header"].attrs["UnitMass_In_CGS"] * u.g
    unit_magnetic_field = 1e4 * u.gauss  # hardcoded right now, can we actually get this from the header???
    return {"Length": unit_length, "Speed": unit_speed, "Mass": unit_mass, "MagneticField": unit_magnetic_field}


def get_snapshot_timeline(output_dir, verbose=False, unit=None, cache_timeline=True):
    """Given a simulation directory, does a pass through the present HDF5 snapshots
    and compiles a list of snapshot paths and their associated

    Parameters
    ----------
    output_dir: string
        Path of the directory containing the snapshots
    verbose: boolean, optional
        Whether to print verbose status updates
    unit: astropy.units.core.PrefixUnit, optional
        Time unit to convert snapshot times to (default: Myr)
    cache_timeline: boolean, optional
        Whether to cache the timeline for future lookup in a file output_dir + "/.timeline" (default: True)

    Returns
    -------
    Tuple containing 2 arrays containing the list of snapshot paths and the corresponding times
    """
    times = []
    snappaths = []
    if verbose:
        print("Getting snapnum timeline...")
    snappaths = natsorted(glob(output_dir + "/snapshot*.hdf5"))
    if not snappaths:
        raise FileNotFoundError(f"No snapshots found in {output_dir}")

    timelinepath = output_dir + "/.timeline"
    if isfile(timelinepath):  # check if we have a cached timeline file
        times = np.load(timelinepath)

    with h5py.File(snappaths[0], "r") as F:
        units = get_snapshot_units(F)

    if len(times) < len(snappaths):
        for f in snappaths:
            with h5py.File(f, "r") as F:
                times.append(F["Header"].attrs["Time"])
    if verbose:
        print("Done!")
    np.save(timelinepath, np.array(times))
    times = np.array(times) * (units["Length"] / units["Speed"]).to(u.Myr)
    dt_units = (times.max() - times.min()).value
    if dt_units > 1000:
        times = times.to(u.Gyr)
    elif dt_units < 0.5:
        times = times.to(u.kyr)
    return np.array(snappaths), times


def multipanel_timelapse_map(
    output_dir=".",
    maps=DEFAULT_MAPS,
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
        # snaps = snappaths[:: len(snappaths) // (times - 1)]
        times = np.linspace(snaptimes.min(), snaptimes.max(), times)  # snaptimes[:: len(snaptimes) // (times - 1)]
    # elif len(times):  # nearest-neighbor snapshots of specified times
    _, ngb_idx = KDTree(np.c_[snaptimes]).query(np.c_[times])
    times = snaptimes[ngb_idx]
    snaps = snappaths[ngb_idx]
    # else:
    # raise NotImplementedError("Format not recognized for supplied times for multipanel map.")

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
