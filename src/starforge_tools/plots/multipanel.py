"""Implements a matplotlib plot consisting of multiple panels, with a timelapse from left to right and
stacking the different maps atop one another.
"""

from natsort import natsorted
from meshoid import Meshoid
from matplotlib import pyplot as plt
import h5py
from glob import glob
from astropy import units as u

DEFAULT_MAPS = ["SurfaceDensity", "Sigma1D", "MassWeightedTemperature", "AlfvenSpeed"]

MINIMAL_DATAFIELDS = set(("PartType0/Masses", "PartType0/Coordinates", "PartType0/SmoothingLength"))
REQUIRED_DATAFIELDS = {
    "SurfaceDensity": MINIMAL_DATAFIELDS,
    "Sigma1D": MINIMAL_DATAFIELDS.union(("PartType0/Velocities",)),
    "MassWeightedTemperature": MINIMAL_DATAFIELDS.union(("PartType0/Temperature",)),
    "AlfvenSpeed": MINIMAL_DATAFIELDS.union(("PartType0/MagneticField", "PartType0/Density")),
}


def get_pdata_for_maps(snapshot_path: str, maps=DEFAULT_MAPS):
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
    unit_mass = F["Header"].attrs["UnitMass_in_CGS"] * u.g
    unit_magnetic_field = 1e4 * u.gauss
    return {"Length": unit_length, "Speed": unit_speed, "Mass": unit_mass, "MagneticField": unit_magnetic_field}


def get_snapshot_timeline(output_dir):
    times = []
    snappaths = []
    print("Getting snapnum timeline...")
    snappaths = natsorted(glob(output_dir + "/snapshot*.hdf5"))
    for f in snappaths:
        with h5py.File(f, "r") as F:
            units = get_snapshot_units(F)
            times.append(F["Header"].attrs["Time"] * units["Length"] / units["Speed"])
    print("Done!")
    return snappaths, times


def multipanel_timelapse_map(maps=DEFAULT_MAPS, times=[0, 3, 6, 9], output_dir="."):
    num_maps, num_times = len(maps), len(times)
    fig, ax = plt.subplots(num_maps, num_times)

    plt.show()

    # get
