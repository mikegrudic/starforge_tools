"""IO routines for multipanel plots"""

from . import rendermaps
from glob import glob
from os.path import isfile
from natsort import natsorted
import h5py
import numpy as np
from astropy import units as u


def get_pdata_for_maps(snapshot_path: str, maps=rendermaps.DEFAULT_MAPS, centering=None) -> dict:
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

    if cache_timeline:
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
