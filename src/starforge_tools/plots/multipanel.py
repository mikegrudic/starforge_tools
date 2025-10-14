"""Implements a matplotlib plot consisting of multiple panels, with a timelapse from left to right and
stacking the different maps atop one another.
"""

from glob import glob
from os.path import isfile
from natsort import natsorted
from matplotlib import pyplot as plt
import h5py
from astropy import units as u
from scipy.spatial import KDTree
import numpy as np
from .map_renderer import MapRenderer

DEFAULT_MAPS = ("SurfaceDensity", "VelocityDispersion", "MassWeightedTemperature", "AlfvenSpeed")

MINIMAL_DATAFIELDS = set(("PartType0/Masses", "PartType0/Coordinates", "PartType0/SmoothingLength"))
REQUIRED_DATAFIELDS = {
    "SurfaceDensity": MINIMAL_DATAFIELDS,
    "VelocityDispersion": MINIMAL_DATAFIELDS.union(("PartType0/Velocities",)),
    "MassWeightedTemperature": MINIMAL_DATAFIELDS.union(("PartType0/Temperature",)),
    "AlfvenSpeed": MINIMAL_DATAFIELDS.union(("PartType0/MagneticField", "PartType0/Density")),
}


def get_pdata_for_maps(snapshot_path: str, maps=DEFAULT_MAPS) -> dict:
    """Does the I/O to get the data required for the specified maps"""
    required_data = set.union(*[REQUIRED_DATAFIELDS[s] for s in maps])
    snapdata = {}
    with h5py.File(snapshot_path, "r") as F:
        for i in range(6):
            s = f"PartType{i}"
            if s not in F.keys():
                continue
            for k in F[s].keys():
                s2 = s + "/" + k
                if s2 in required_data:
                    snapdata[s2] = F[s2][:]

    return snapdata


def get_snapshot_units(F):
    unit_length = F["Header"].attrs["UnitLength_In_CGS"] * u.cm
    unit_speed = F["Header"].attrs["UnitVelocity_In_CGS"] * u.cm / u.s
    unit_mass = F["Header"].attrs["UnitMass_In_CGS"] * u.g
    unit_magnetic_field = 1e4 * u.gauss
    return {"Length": unit_length, "Speed": unit_speed, "Mass": unit_mass, "MagneticField": unit_magnetic_field}


def get_snapshot_timeline(output_dir, verbose=False):
    print(output_dir)
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
    return np.array(snappaths), np.array(times) * (units["Length"] / units["Speed"]).to(u.Myr)


def multipanel_timelapse_map(output_dir=".", maps=DEFAULT_MAPS, times=4, res=1024, length=20):
    snappaths, snaptimes = get_snapshot_timeline(output_dir)

    if isinstance(times, int):  # if we specified an integer number of times, assume evenly-spaced
        snaps = snappaths[:: len(snappaths) // (times - 1)]
        times = snaptimes[:: len(snaptimes) // (times - 1)]
    elif len(times):  # eventually implement nearest-neighbor snapshots of specified times
        _, ngb_idx = KDTree(np.c_[snaptimes]).query(np.c_[times])
        snaps = snappaths[ngb_idx]
    else:
        raise NotImplementedError("Format not recognized for supplied times for multipanel map.")

    num_maps, num_times = len(maps), len(times)
    fig, ax = plt.subplots(num_maps, num_times, figsize=(8, 8))
    mapargs = {"size": length, "res": res}
    X, Y = 2 * [np.linspace(-0.5 * length, 0.5 * length, res)]
    for i in range(len(times)):
        pdata = get_pdata_for_maps(snaps[i], maps)
        renderer = MapRenderer(pdata, mapargs)
        for j, map_name in enumerate(maps):
            # print(i, j)
            map = renderer.get_map(map_name)
            ax[j, i].pcolormesh(X, Y, np.log10(map))

    plt.savefig("multipanel.png")
