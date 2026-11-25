"""Projects the per-cell H2 formation and destruction rates written by h2_formation_destruction.py into maps, with
the formation rate also split between jet material (jet mass fraction >= 0.5) and the rest, each with and without
the clumping factor.

Usage: python dump_maps.py [simulation output directory]"""

from sys import argv
from glob import glob
from natsort import natsorted
import numpy as np
from meshoid import Meshoid
import h5py

DEFAULT_OUTPUT_DIR = "/mnt/ceph/users/starforge/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output"
FEEDBACK_TRACER_KEY = -3  # header key of the Metallicity columns tracing jets, winds and SNe, in that order
RES = 1024
RMAX = 20  # half-width of the maps in code length units (pc)


def dump_map(rates_path):
    snapnum = rates_path.split("_")[-1].split(".hdf5")[0]
    snapshot_path = rates_path.split("/H2_formation_destruction/")[0] + f"/snapshot_{snapnum}.hdf5"

    with h5py.File(rates_path) as F:
        particle_ids = F["ParticleIDs"][:]
        J_F = {"": F["H2_Formation_Rate_Msun_Myr"][:], "_noclump": F["H2_Formation_Rate_NoClumping_Msun_Myr"][:]}
        J_D = F["H2_Destruction_Rate_Msun_Myr"][:]
        time_Myr = F["Time_Myr"][()]

    with h5py.File(snapshot_path) as F:
        boxsize = F["Header"].attrs["BoxSize"]
        jet_tracer = int(np.flatnonzero(F["Header"].attrs["Metals_Atomic_Number_Or_Key"] == FEEDBACK_TRACER_KEY)[0])
        coordinates = F["PartType0/Coordinates"][:]
        # the cells h2_formation_destruction.py kept, in its order. ParticleIDs are not unique, so cannot select on them
        radial_cut = np.sum((coordinates - 0.5 * boxsize) ** 2, axis=1) ** 0.5 < boxsize * 0.4
        if not np.array_equal(F["PartType0/ParticleIDs"][:][radial_cut], particle_ids):
            raise ValueError(f"cells of {rates_path} do not match those selected from {snapshot_path}")
        coordinates = coordinates[radial_cut] - 0.5 * boxsize
        masses = np.float64(F["PartType0/Masses"][:][radial_cut])
        smoothing_length = np.float64(F["PartType0/SmoothingLength"][:][radial_cut])
        f_jet = F["PartType0/Metallicity"][:, jet_tracer][radial_cut]

    M = Meshoid(coordinates, masses, smoothing_length)
    X = Y = np.linspace(-RMAX, RMAX, RES)
    X, Y = np.meshgrid(X, Y, indexing="ij")
    args = {"size": 2 * RMAX, "res": RES, "center": np.zeros(3)}
    is_jet = f_jet >= 0.5
    sigma_JF_jet = {k: M.SurfaceDensity(np.where(is_jet, J, 0), **args) for k, J in J_F.items()}
    sigma_JF_nonjet = {k: M.SurfaceDensity(np.where(is_jet, 0, J), **args) for k, J in J_F.items()}
    sigma_JD = M.SurfaceDensity(J_D, **args)
    sigma_gas = M.SurfaceDensity(masses, **args)

    with h5py.File(rates_path.replace(".hdf5", "_MAP.hdf5"), "w") as F:
        for k in J_F:  # "" with the clumping factor, "_noclump" without
            F.create_dataset(f"Sigma_JF{k}_Msun_Myr_pc2", data=sigma_JF_jet[k] + sigma_JF_nonjet[k])
            F.create_dataset(f"Sigma_JF{k}_jet_Msun_Myr_pc2", data=sigma_JF_jet[k])
            F.create_dataset(f"Sigma_JF{k}_nonjet_Msun_Myr_pc2", data=sigma_JF_nonjet[k])
        F.create_dataset("Sigma_JD_Msun_Myr_pc2", data=sigma_JD)
        F.create_dataset("Sigma_gas_Msun_pc2", data=sigma_gas)
        F.create_dataset("X_pc", data=X)
        F.create_dataset("Y_pc", data=Y)
        F.create_dataset("Time_Myr", data=time_Myr)


def main():
    output_dir = argv[1] if len(argv) > 1 else DEFAULT_OUTPUT_DIR
    rates_paths = glob(output_dir + "/H2_formation_destruction/H2_formation_destruction_*.hdf5")
    for rates_path in natsorted(f for f in rates_paths if not f.endswith("_MAP.hdf5")):
        print(rates_path)
        dump_map(rates_path)


if __name__ == "__main__":
    main()
