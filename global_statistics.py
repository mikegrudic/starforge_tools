"""Does pass on the simulations to compute desired global statistics as a function of time"""

from os.path import isfile
from glob import glob
import numpy as np
import h5py
import astropy.units as u
from joblib import Parallel, delayed
from natsort import natsorted
from astropy.table import Table
from astropy.io import ascii


class GlobalStatistics:
    """
    Class that takes an hdf5 file object as input and automatically computes
    any global statistic that has a defined 'get_<quantity name>' method.

    To add a new quantity, just add a new get_ method to this class and it will
    automatically compute it and store the result.
    """

    def __init__(self, snapfile: h5py.File, stats_to_compute=None):
        self.snapfile = snapfile
        self.header = self.snapfile["Header"].attrs

        self.stored_datafields = []
        for f in dir(self):
            # loop over callable methods
            if callable(getattr(self, f)) and "get_" in f:
                datafield = f.replace("get_", "")
                if (stats_to_compute is not None) and (
                    not datafield in stats_to_compute
                ):
                    continue
                setattr(self, datafield, getattr(self, f)())  # call the getter function
                if datafield != "units":
                    self.stored_datafields.append(datafield)

    def get_units(self):
        """Returns a dict specifying the units system of the snapshot"""
        unitdict = {}
        for k in "UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS":
            unitdict[k] = self.header[k]
        unitdict["UnitTime_In_CGS"] = (
            unitdict["UnitLength_In_CGS"] / unitdict["UnitVelocity_In_CGS"]
        )
        return unitdict

    def get_stellar_mass_sum(self):
        """Returns the total stellar mass in code units."""
        if "PartType5" in self.snapfile.keys():
            return np.sum(self.snapfile["PartType5/BH_Mass"][:])
        return 0

    def get_stellar_mass_max(self):
        """Returns the maximum stellar mass in code units."""
        if "PartType5" in self.snapfile.keys():
            return np.max(self.snapfile["PartType5/BH_Mass"][:])
        return 0

    def get_time_myr(self):
        """Returns the snapshot time in Myr"""
        unit_dict = self.get_units()
        return self.header["Time"] * unit_dict["UnitTime_In_CGS"] * u.second.to(u.Myr)


def get_stats_from_snapshot(snappath, stats_to_compute=None):
    """Returns a global statistics instance containing computed global statistics"""
    with h5py.File(snappath, "r") as f:
        return GlobalStatistics(f, stats_to_compute)


def statslist_to_table(statslist):
    """Convert a list of GlobalStatistics instances to an astropy table"""
    fields = statslist[0].stored_datafields
    statsdict = {f: [] for f in fields}
    for s in statslist:
        for f, l in statsdict.items():
            statsdict[f].append(getattr(s, f))
    tab = Table()
    for f in fields:
        tab[f] = statsdict[f]
    return tab


def get_globalstats_of_simulation(
    rundir, n_jobs=-1, stats_to_compute=None, overwrite=False
):
    """Does a pass over all snapshots to compute desired global statistics"""
    datafile_path = rundir + "/global_statistics.dat"
    if isfile(datafile_path) and not overwrite:
        tab = Table.read(datafile_path, format="ascii.basic")
        if stats_to_compute is not None:
            if set(stats_to_compute).intersection(tab.keys()) == stats_to_compute:
                return tab
        else:
            return tab

    snaps = glob(rundir + "/snapshot*.hdf5") + glob(
        rundir + "/stars_only/snapshot*.hdf5"
    )
    snaps = natsorted(snaps)
    if n_jobs == 1:
        stats = list(map(get_stats_from_snapshot, snaps))
    else:
        stats = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(get_stats_from_snapshot)(s, stats_to_compute) for s in snaps
        )
    # ok now convert to table format
    tab = statslist_to_table(stats)
    ascii.write(tab, datafile_path, format="basic", overwrite=True)
    return tab
