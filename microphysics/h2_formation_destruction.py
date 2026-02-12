"""Given starforge snapshots, computes the formation and destruction rates of H2 and outputs an HDF5 file with corresponding values
for each gas cell."""

import h5py
from os.path import isdir
from os import mkdir
from joblib import delayed, Parallel
from sys import argv
from jaco.models.starforge.H2_chemistry.grain_formation import grain_formation
import sympy as sp


def output_dir_path(snapshot_path: str) -> str:
    return snapshot_path.split("/snapshot")[0] + "/H2_formation_destruction"


def H2_formation_rate(T=None, Td=None, Z_d=None, f_d=None, n_H=None):
    """Returns the H2 formation rate in cm^-3 s^-1"""
    rate = grain_formation.rate

    func = sp.lambdify(sp.symbols("T Td Z_d f_d n_H"), rate, "jax")
    return func(T, Td, Z)


def compute_h2_rates(snapshot_path: str):
    """Computes the formation and destruction rates for gas in the snapshot referenced by snapshot_path, and writes
    an HDF5 file in the directory H2_formation_destruction next to the snapshot
    """
    if ".hdf5" not in snapshot_path:
        return
    output_dir = output_dir_path(snapshot_path)

    if not isdir(output_dir):
        mkdir(output_dir)

    snapnum = snapshot_path.split(".hdf5")[0].split("_")[-1]
    output_filepath = output_dir + f"/H2_formation_destruction_{snapnum}.hdf5"

    # with h5py.File(output_filepath, "w") as F:
    # return


def main():
    snappaths = argv[1:]
    Parallel(n_jobs=-1)(delayed(compute_h2_rates)(s) for s in snappaths)


if __name__ == "__main__":
    main()
