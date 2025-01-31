"""Does pass on the simulations to compute desired global statistics as a function of time"""

from os.path import isfile
from glob import glob
import numpy as np
import h5pickle as h5py
from astropy import units as u, constants as c
from joblib import Parallel, delayed, wrap_non_picklable_objects
from multiprocessing import Pool
from natsort import natsorted
from astropy.table import QTable
from starforge_tools.star_properties import Q_ionizing, wind_mdot, vwind


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
                if (stats_to_compute is not None) and (not datafield in stats_to_compute):
                    continue
                setattr(self, datafield, getattr(self, f)())  # call the getter function
                if datafield != "units":
                    self.stored_datafields.append(datafield)

        del self.snapfile
        del self.header

    def get_units(self):
        """Returns a dict specifying the units system of the snapshot"""
        unitdict = {}
        for k in "UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS":
            unitdict[k] = self.header[k]
        unitdict["UnitTime_In_CGS"] = unitdict["UnitLength_In_CGS"] / unitdict["UnitVelocity_In_CGS"]
        return unitdict

    def get_stellar_mass_sum(self):
        """Returns the total stellar mass in code units."""
        if "PartType5" in self.snapfile.keys():
            return np.sum(self.snapfile["PartType5/BH_Mass"][:]) * u.M_sun
        return 0 * u.M_sun

    def get_stellar_mass_max(self):
        """Returns the maximum stellar mass in code units."""
        if "PartType5" in self.snapfile.keys():
            return np.max(self.snapfile["PartType5/BH_Mass"][:]) * u.M_sun
        return 0 * u.M_sun

    def get_bolometric_luminosity(self):
        """Returns the total bolometric luminosity in solar units."""
        if "PartType5" in self.snapfile.keys():
            return np.sum(self.snapfile["PartType5/StarLuminosity_Solar"][:]) * u.L_sun
        return 0 * u.L_sun

    def get_ionizing_flux(self):
        """Returns the number of H-ionizing photons emitted per second."""
        if "PartType5" in self.snapfile.keys():
            L = self.snapfile["PartType5/StarLuminosity_Solar"][:]
            R = self.snapfile["PartType5/ProtoStellarRadius_inSolar"][:]
            Q = Q_ionizing(lum=L, radius=R) / u.s
            return Q.sum()
        return 0 / u.s

    def get_wind_mdot(self):
        """Returns the total wind mass loss rate in solar mass/yr."""
        if "PartType5" in self.snapfile.keys():
            L = self.snapfile["PartType5/StarLuminosity_Solar"][:]
            M = self.snapfile["PartType5/BH_Mass"][:]
            Z = self.snapfile["PartType5/Metallicity"][:, 0] / 0.014
            mdot = wind_mdot(M, L, Z)
            return mdot.sum() * u.M_sun / u.yr
        return 0 * u.M_sun / u.yr

    def get_wind_luminosity(self):
        """Returns the total wind luminosity in cgs."""
        if "PartType5" in self.snapfile.keys():
            L = self.snapfile["PartType5/StarLuminosity_Solar"][:]
            R = self.snapfile["PartType5/ProtoStellarRadius_inSolar"][:]
            M = self.snapfile["PartType5/BH_Mass"][:]
            Z = self.snapfile["PartType5/Metallicity"][:, 0] / 0.014
            mdot = wind_mdot(M, L, Z) * u.M_sun / u.yr
            vw = vwind(M, L, R) * u.km / u.s
            Lwind_cgs = (0.5 * mdot * vw * vw).cgs
            return Lwind_cgs.sum()
        return 0 * u.erg / u.s

    def get_time(self):
        """Returns the snapshot time in Myr"""
        unit_dict = self.get_units()
        return (self.header["Time"] * unit_dict["UnitTime_In_CGS"] * u.second).to(u.Myr)


# @delayed
# @wrap_non_picklable_objects
def get_stats_from_snapshot(snappath, stats_to_compute=None):
    print(snappath)
    """Returns a global statistics instance containing computed global statistics"""
    with h5py.File(snappath, "r") as f:
        stats = GlobalStatistics(f, stats_to_compute)
    return stats


def statslist_to_table(statslist):
    """Convert a list of GlobalStatistics instances to an astropy table"""
    fields = statslist[0].stored_datafields
    statsdict = {f: [] for f in fields}
    for s in statslist:
        for f, l in statsdict.items():
            statsdict[f].append(getattr(s, f))
    tab = QTable()
    for f in fields:
        tab[f] = statsdict[f]
    return tab


def get_globalstats_of_simulation(rundir, stats_to_compute=None, n_jobs=-1, overwrite=False, stars_only_snaps=True):
    """Does a pass over all snapshots to compute desired global statistics"""
    datafile_path = rundir + "/global_statistics.fits"
    format = "fits"
    if isfile(datafile_path) and not overwrite:
        tab = Table.read(datafile_path, format=format)  # , format="ascii.basic")
        if stats_to_compute is not None:
            if set(stats_to_compute).intersection(tab.keys()) == stats_to_compute:
                return tab
        else:
            return tab

    snaps = glob(rundir + "/snapshot*.hdf5")
    if stars_only_snaps:
        snaps += glob(rundir + "/stars_only/snapshot*.hdf5")
    snaps = natsorted(snaps)
    if n_jobs == 1:
        stats = list(map(get_stats_from_snapshot, snaps))
    else:
        #        stats = Pool().starmap(get_stats_from_snapshot, zip(snaps, len(snaps) * [stats_to_compute]))
        stats = Parallel(n_jobs=n_jobs)(delayed(get_stats_from_snapshot)(s, stats_to_compute) for s in snaps)
    # ok now convert to table format
    #    print(stats)
    tab = statslist_to_table(stats)
    tab = tab[tab["time"].argsort()]
    tab.write(datafile_path, overwrite=True, format=format)
    return tab
