"""Given starforge snapshots, computes the formation and destruction rates of H2 and outputs an HDF5 file with corresponding values
for each gas cell.

The microphysics (rates, clumping factor, dust survival, constants and abundance conventions) is jaco's; this script
only converts the snapshot to jaco's inputs."""

import h5py
from os import makedirs
from joblib import delayed, Parallel
from sys import argv
from jaco.data import SolarAbundances
from jaco.models.starforge.h2_chemistry.grain_formation import grain_formation
from jaco.models.starforge.h2_chemistry.photochemistry import photodissociation
from jaco.models.starforge.h2_chemistry.cosmic_ray_dissociation import cosmic_ray_dissociation
from jaco.models.starforge.symbols import MASS_NUMBER, PROTONMASS_CGS, T_dust, clumping_factor
from jaco.models.starforge.dust_opacity import dust_survival
from jaco.models.starforge.radiation import U_HABING
import sympy as sp
import numpy as np
from meshoid import Meshoid
from astropy import units as u

HE_ATOMIC_NUMBER, FE_ATOMIC_NUMBER = 2, 26  # keys of the Metallicity columns in the snapshot header


def output_dir_path(snapshot_path: str) -> str:
    return snapshot_path.split("/snapshot")[0] + "/H2_formation_destruction"


def evaluate(expression, inputs: dict):
    """Evaluates a jaco expression on the arrays in inputs, which are keyed by symbol name and in the cgs units jaco
    assumes"""
    symbols = sorted(expression.free_symbols, key=str)
    missing = [str(s) for s in symbols if str(s) not in inputs]
    if missing:
        raise KeyError(f"jaco expression needs symbols that were not provided: {missing}")
    return sp.lambdify(symbols, expression, "numpy")(*(inputs[str(s)] for s in symbols))


def H2_formation_rate(inputs: dict, clumping=True):
    """Returns the rate of H2 formation on dust grains in cm^-3 s^-1, with or without GIZMO's clumping factor"""
    reaction = grain_formation()
    if clumping:
        return evaluate(reaction.rate.subs(reaction.clumping, clumping_factor), inputs)
    return evaluate(reaction.unclumped().rate, inputs)


def H2_destruction_rate(inputs: dict):
    """Returns the rate of H2 photodissociation plus cosmic-ray dissociation in cm^-3 s^-1"""
    return evaluate(photodissociation("H_2").rate + cosmic_ray_dissociation("H_2").rate, inputs)


def compute_h2_rates(snapshot_path: str):
    """Computes the formation and destruction rates for gas in the snapshot referenced by snapshot_path, and writes
    an HDF5 file in the directory H2_formation_destruction next to the snapshot
    """
    print(snapshot_path)
    if ".hdf5" not in snapshot_path:
        return
    output_dir = output_dir_path(snapshot_path)
    makedirs(output_dir, exist_ok=True)

    snapnum = snapshot_path.split(".hdf5")[0].split("_")[-1]
    output_filepath = output_dir + f"/H2_formation_destruction_{snapnum}.hdf5"

    fields_to_load = (
        "Masses",
        "Density",
        "Coordinates",
        "SmoothingLength",
        "Temperature",
        "Dust_Temperature",
        "MolecularMassFraction",
        "NeutralHydrogenAbundance",
        "Velocities",
    )

    with h5py.File(snapshot_path, "r") as F:
        header = dict(F["Header"].attrs)
        boxsize = header["BoxSize"]
        r = np.sum((F["PartType0/Coordinates"][:] - 0.5 * boxsize) ** 2, axis=1) ** 0.5
        radial_cut = r < boxsize * 0.4
        # snapshots are single precision, which cgs volumes overflow
        pdata = {f: np.float64(F["PartType0/" + f][:][radial_cut]) for f in fields_to_load}
        particle_ids = F["PartType0/ParticleIDs"][:][radial_cut]
        metal_keys = header["Metals_Atomic_Number_Or_Key"]
        helium, iron = (int(np.flatnonzero(metal_keys == k)[0]) for k in (HE_ATOMIC_NUMBER, FE_ATOMIC_NUMBER))
        Z, Z_He, Z_Fe = (np.float64(F["PartType0/Metallicity"][:, k][radial_cut]) for k in (0, helium, iron))
        E_FUV = np.float64(F["PartType0/PhotonEnergy"][:, 1][radial_cut])  # FUV band

    unit_length, unit_mass, unit_velocity = (header[f"Unit{q}_In_CGS"] for q in ("Length", "Mass", "Velocity"))
    time_Myr = header["Time"] * (unit_length / unit_velocity * u.s).to_value(u.Myr)

    density = pdata["Density"] * unit_mass / unit_length**3
    volume = pdata["Masses"] / pdata["Density"] * unit_length**3
    X_H = 1 - Z - Z_He
    x_H0 = pdata["NeutralHydrogenAbundance"]
    f_H2 = pdata["MolecularMassFraction"]  # mass fraction of the neutral H that is molecular
    n_Htot = X_H * density / PROTONMASS_CGS
    n_H = x_H0 * (1 - f_H2) * n_Htot
    n_H2 = 0.5 * x_H0 * f_H2 * n_Htot

    M = Meshoid(pdata["Coordinates"])
    column_density = (
        0.5 * pdata["SmoothingLength"] * pdata["Density"]
        + pdata["Density"] ** 2 / np.sum(M.D(pdata["Density"]) ** 2, axis=1) ** 0.5
    )

    inputs = {
        "T": pdata["Temperature"],
        "Td": pdata["Dust_Temperature"],
        "Z_d": Z / SolarAbundances.mass_fraction["Z"],
        "n_Htot": n_Htot,
        "n_H": n_H,
        "n_H_2": n_H2,
        "x_H_2": n_H2 / n_Htot,
        "x_Fe": Z_Fe / (MASS_NUMBER["Fe"] * X_H),
        "G_LW": E_FUV * unit_mass * unit_velocity**2 / volume / U_HABING,
        "N_H": column_density * unit_mass / unit_length**2 / PROTONMASS_CGS,
        "Δx": volume ** (1.0 / 3),
        "∇v": np.sqrt(np.sum(M.D(pdata["Velocities"]) ** 2, axis=(1, 2))) * unit_velocity / unit_length,
        "ISRF": header["InterstellarRadiationFieldStrength"],
    }
    inputs["f_d"] = evaluate(dust_survival(T_dust), inputs)

    # cm^-3 s^-1 of H2 molecules to Msun/Myr of H2
    to_Msun_Myr = volume * 2 * PROTONMASS_CGS * (u.g / u.s).to(u.Msun / u.Myr)
    form = (H2_formation_rate(inputs) * to_Msun_Myr).clip(0)
    form_unclumped = (H2_formation_rate(inputs, clumping=False) * to_Msun_Myr).clip(0)
    destruction = H2_destruction_rate(inputs) * to_Msun_Myr

    with h5py.File(output_filepath, "w") as F:
        F.create_dataset("ParticleIDs", data=particle_ids)
        F.create_dataset("G0", data=inputs["G_LW"])
        F.create_dataset("n_H", data=n_H)
        F.create_dataset("ColumnDensity", data=inputs["N_H"])
        F.create_dataset("ClumpingFactor", data=evaluate(clumping_factor, inputs))
        F.create_dataset("H2_Formation_Rate_Msun_Myr", data=form)
        F.create_dataset("H2_Formation_Rate_NoClumping_Msun_Myr", data=form_unclumped)
        F.create_dataset("H2_Destruction_Rate_Msun_Myr", data=destruction)
        F.create_dataset("H2_Formation_Rate_total", data=np.sum(form))
        F.create_dataset("H2_Destruction_Rate_total", data=np.sum(destruction))
        F.create_dataset("Time_Myr", data=time_Myr)

    return


def main():
    snappaths = argv[1:]
    Parallel(n_jobs=2)(delayed(compute_h2_rates)(s) for s in snappaths)
    # [compute_h2_rates(s) for s in snappaths]


if __name__ == "__main__":
    main()
