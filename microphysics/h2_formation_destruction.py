"""Given starforge snapshots, computes the formation and destruction rates of H2 and outputs an HDF5 file with corresponding values
for each gas cell."""

import h5py
from os.path import isdir
from os import mkdir
from joblib import delayed, Parallel
from sys import argv
from jaco.models.starforge.h2_chemistry.grain_formation import grain_formation
from jaco.models.starforge.h2_chemistry.photochemistry import photodissociation
from jaco.models.starforge.h2_chemistry.cosmic_ray_dissociation import cosmic_ray_dissociation
import sympy as sp
import numpy as np
from meshoid import Meshoid
from astropy import units as u, constants as c


def output_dir_path(snapshot_path: str) -> str:
    return snapshot_path.split("/snapshot")[0] + "/H2_formation_destruction"


def H2_formation_rate(T, Td, Z_d, f_d, n_H, C_2=1.0):
    """Returns the H2 formation rate in cm^-3 s^-1"""
    formation_rate = grain_formation.rate
    rate_lambdified = sp.lambdify(list(sp.ordered(formation_rate.free_symbols)), formation_rate, "numpy")
    return rate_lambdified(T=T, Td=Td, Z_d=Z_d, f_d=f_d, n_H=n_H, C_2=C_2) / u.s / u.cm**3


def H2_dissociation_rate(G_0, N_H, T, n_H, n_H_2, x_Hplus, dx, gradv_kms_pc, ISRF=1.0):
    """Returns the dissociation rate of an H_2 molecule in s^-1"""
    rate = photodissociation("H_2").rate + cosmic_ray_dissociation("H_2").rate
    rate = rate.subs("x_H+", "x_Hplus").subs("Δx", "dx").subs("∇v", "gradv").subs("ISRF", ISRF)
    f_lambified = sp.lambdify(list(sp.ordered(rate.free_symbols)), rate, "numpy")
    return (
        f_lambified(G_0=G_0, N_H=N_H, T=T, n_H=n_H, n_H_2=n_H_2, x_Hplus=x_Hplus, dx=dx, gradv=gradv_kms_pc)
        / u.s
        / u.cm**3
    )


def u_FUV_to_habing(u_FUV_code, unit_mass=u.Msun, unit_length=u.pc, unit_vel=u.m / u.s):
    """Convert between FUV energy density in code units and Habing flux units"""
    habing_flux_cgs = 1.6e-3
    FUV_flux_cgs = (u_FUV_code * unit_vel**2 * unit_mass / unit_length**3 * c.c).to(u.erg / u.cm**2 / u.s)
    return FUV_flux_cgs.value / habing_flux_cgs


def compute_h2_rates(snapshot_path: str):
    """Computes the formation and destruction rates for gas in the snapshot referenced by snapshot_path, and writes
    an HDF5 file in the directory H2_formation_destruction next to the snapshot
    """
    print(snapshot_path)
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
        "Velocities",
        "ParticleIDs",
        "SoundSpeed",
    )

    # print("Doing IO...")
    with h5py.File(snapshot_path, "r") as F:
        boxsize = F["Header"].attrs["BoxSize"]
        r = np.sum((F["PartType0/Coordinates"][:] - 0.5 * boxsize) ** 2, axis=1) ** 0.5
        radial_cut = r < boxsize * 0.4
        pdata = {f: F["PartType0/" + f][:][radial_cut] for f in fields_to_load}
        time_Myr = F["Header"].attrs["Time"] * (u.pc / (u.m / u.s)).to(u.Myr)

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
    G0 = u_FUV_to_habing(u_FUV)

    M = Meshoid(pdata["Coordinates"])
    column_density = (
        0.5 * pdata["SmoothingLength"] * pdata["Density"]
        + pdata["Density"] ** 2 / np.sum(M.D(pdata["Density"]) ** 2, axis=1) ** 0.5
    )
    gradv_kms_pc = np.sqrt(np.sum(M.D(pdata["Velocities"] / 1e3) ** 2, axis=(1, 2)))
    vol = pdata["Masses"] / pdata["Density"]
    dx = vol ** (1.0 / 3)
    vturb = gradv_kms_pc * dx
    mach = vturb / (0.111 * pdata["Temperature"] ** 0.5 / 3**0.5)
    C_2 = 1 + (0.5 * mach) ** 2

    form = (H2_formation_rate(T, Td, Z_d, f_d, n_H, C_2) * volume * u.pc**3 * (2 * c.m_p)).to(u.Msun / u.Myr)
    form = form.clip(
        0,
    )
    total_H2_form_mass = np.sum(form)

    N_H = (column_density * u.Msun / u.pc**2 * X_H / c.m_p).to(u.cm**-2).value

    destruction = (
        H2_dissociation_rate(G0, N_H, T, n_H, n_H2, x_Hplus, dx, gradv_kms_pc) * volume * u.pc**3 * 2 * c.m_p
    ).to(u.Msun / u.Myr)

    with h5py.File(output_filepath, "w") as F:
        F.create_dataset("ParticleIDs", data=pdata["ParticleIDs"])
        F.create_dataset("G0", data=G0)
        F.create_dataset("n_H", data=n_H)
        F.create_dataset("ColumnDensity", data=N_H)
        F.create_dataset("H2_Formation_Rate_Msun_Myr", data=form)
        F.create_dataset("Mach", data=mach)
        F.create_dataset("H2_Destruction_Rate_Msun_Myr", data=destruction)
        F.create_dataset("H2_Formation_Rate_total", data=total_H2_form_mass)
        F.create_dataset("H2_Destruction_Rate_total", data=np.sum(destruction))
        F.create_dataset("Time_Myr", data=time_Myr)

    return


def main():
    snappaths = argv[1:]
    Parallel(n_jobs=1)(delayed(compute_h2_rates)(s) for s in snappaths)
    # [compute_h2_rates(s) for s in snappaths]


if __name__ == "__main__":
    main()
