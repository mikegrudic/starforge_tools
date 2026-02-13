"""Given starforge snapshots, computes the formation and destruction rates of H2 and outputs an HDF5 file with corresponding values
for each gas cell."""

import h5py
from os.path import isdir
from os import mkdir
from joblib import delayed, Parallel
from sys import argv
from jaco.models.starforge.h2_chemistry.grain_formation import grain_formation
from jaco.models.starforge.h2_chemistry.photochemistry import f_selfshield_H2
import sympy as sp
import numpy as np
from meshoid import Meshoid
from astropy import units as u, constants as c


def output_dir_path(snapshot_path: str) -> str:
    return snapshot_path.split("/snapshot")[0] + "/H2_formation_destruction"


def H2_formation_rate(T, Td, Z_d, f_d, n_H):
    """Returns the H2 formation rate in cm^-3 s^-1"""
    formation_rate = grain_formation.rate
    rate_lambdified = sp.lambdify(list(sp.ordered(formation_rate.free_symbols)), formation_rate, "numpy")
    return rate_lambdified(T=T, Td=Td, Z_d=Z_d, f_d=f_d, n_H=n_H) * u.cm**-3 / u.s


def f_H2_selfshield(N_H, T, n_H, n_H_2, x_Hplus, dx, gradv_kms_pc):
    f = f_selfshield_H2().subs("H+", "Hplus").subs("Δx", "dx").subs("∇v", "gradv")
    f_lambified = sp.lambdify(list(sp.lambdify(f.free_symbols)), f, "numpy")
    return f_lambified(N_H=N_H, T=T, n_H=n_H, n_H_2=n_H_2, x_Hplus=x_Hplus, dx=dx, gradv=gradv)


# def H2_dissociation_rate(T,Td,Z_d,f_d,n_H):


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

    fields_to_load = (
        "Masses",
        "Density",
        "Coordinates",
        "SmoothingLength",
        "Metallicity",
        "Temperature",
        "Dust_Temperature",
        "MolecularMassFraction",
        "NeutralHydrogenAbundance",
        "PhotonEnergy",
    )

    with h5py.File(snapshot_path, "r") as F:
        pdata = {f: F["PartType0/" + f][:] for f in fields_to_load}

    volume = pdata["Masses"] / pdata["Density"]
    X_H = 0.74
    Z_d = pdata["Metallicity"][:, 0] / 0.0142
    f_d = np.ones_like(Z_d)
    T = pdata["Temperature"]
    Td = pdata["Dust_Temperature"]
    x_Hplus = 1 - pdata["NeutralHydrogenAbundance"]
    n_H = (
        pdata["NeutralHydrogenAbundance"]
        * pdata["Density"]
        * (u.Msun / u.pc**3)
        * (1 - pdata["MolecularMassFraction"])
        * X_H
        / c.m_p
    ).cgs.value  # we add the units in the jaco function
    n_H2 = (
        pdata["NeutralHydrogenAbundance"]
        * pdata["Density"]
        * (u.Msun / u.pc**3)
        * pdata["MolecularMassFraction"]
        * X_H
        / (2 * c.m_p)
    ).cgs.value  # we add the units in the jaco function
    u_FUV = pdata["PhotonEnergy"][:, 1] / volume
    G0 = u_FUV

    M = Meshoid(pdata["Coordinates"])
    column_density = (
        0.5 * pdata["SmoothingLength"] * pdata["Density"]
        + pdata["Density"] ** 2 / np.sum(M.D(pdata["Density"]) ** 2, axis=1) ** 0.5
    )
    gradv = np.sqrt(np.sum(M.D(pdata["Velocities"]) ** 2, axis=(1, 2)))

    form = (H2_formation_rate(T, Td, Z_d, f_d, n_H) * volume * u.pc**3 * (2 * c.m_p)).to(u.Msun / u.Myr)
    total_H2_form_mass = np.sum(form)

    with h5py.File(output_filepath, "w") as F:
        F.create_dataset("H2_Formation_Rate_Msun_Myr", data=form)
        F.create_dataset("H2_Formation_Rate_total", data=total_H2_form_mass)

    return


def main():
    snappaths = argv[1:]
    #    Parallel(n_jobs=-1)(delayed(compute_h2_rates)(s) for s in snappaths)
    [compute_h2_rates(s) for s in snappaths]


if __name__ == "__main__":
    main()
