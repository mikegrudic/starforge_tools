#!/usr/bin/env python

"""                                                                            
Adds a single star to specified snapshots.

Usage: AddStarToSnapshot.py <files> ... [options]

Options:                                                                       
   -h --help                  Show this screen.
   --pos=<X,Y,Z>              Choose from specifying star coordinates, "COM" for center of mass, or "densest" for point of maximum density [default: 0,0,0]
   --M=<msun>                 Stellar mass [default: 40]
   --v=<VX,VY,VZ>             Choose from specifying comma-separated velocities in km/s or "comoving" to use local gas velocity [default: 0,0,0]
   --age=<Myr>                Stellar age in Myr [default: 0]
"""

from sys import argv
import h5py
from os.path import isdir, abspath, isfile
from os import mkdir, system
from matplotlib import pyplot as plt
import numpy as np
from docopt import docopt
from scipy.spatial.distance import cdist

options = docopt(__doc__)
snapshot_paths = options["<files>"]
M = float(options["--M"])


def luminosity_tout(ms):
    """Fit of main-sequence luminosity as a function of main-sequence mass from Tout 1996MNRAS.281..257T, in solar units"""
    L_ms = (0.39704170 * np.power(ms, 5.5) + 8.52762600 * np.power(ms, 11)) / (
        0.00025546
        + np.power(ms, 3)
        + 5.43288900 * np.power(ms, 5)
        + 5.56357900 * np.power(ms, 7)
        + 0.78866060 * np.power(ms, 8)
        + 0.00586685 * np.power(ms, 9.5)
    )
    L_ms = np.atleast_1d(L_ms)
    L_ms[np.isnan(L_ms)] = 0.0
    return L_ms


def radius_tout(ms):
    """Fit of main-sequence radius as a function of main-sequence mass from Tout 1996MNRAS.281..257T, in solar units"""
    R_ms = (
        1.71535900 * np.power(ms, 2.5)
        + 6.59778800 * np.power(ms, 6.5)
        + 10.08855000 * np.power(ms, 11)
        + 1.01249500 * np.power(ms, 19)
        + 0.07490166 * np.power(ms, 19.5)
    ) / (
        0.01077422
        + 3.08223400 * np.power(ms, 2)
        + 17.84778000 * np.power(ms, 8.5)
        + np.power(ms, 18.5)
        + 0.00022582 * np.power(ms, 19.5)
    )
    R_ms = np.atleast_1d(R_ms)
    R_ms[np.isnan(R_ms)] = 0.0
    return R_ms


# print(luminosity_tout(40.))

for f in snapshot_paths:
    fname = f.replace(".hdf5", "_star.hdf5")
    if isfile(fname):
        system("rm " + fname)
    Fstar = h5py.File(fname, "w")
    with h5py.File(f, "r") as F:
        for k in F.keys():
            F.copy(k, Fstar)

        boxsize = F["Header"].attrs["BoxSize"]
        Fstar.create_group("PartType5")
        Ngas = len(np.array(Fstar["PartType0/Coordinates"][:]))
        if options["--pos"] == "COM":
            xstar = np.average(Fstar["PartType0/Coordinates"][:], axis=0)
        elif options["--pos"] == "densest":
            xstar = Fstar["PartType0/Coordinates"][:][
                Fstar["PartType0/Density"][:].argmax()
            ]
        else:
            xstar = (
                np.array([float(s) for s in options["--pos"].split(",")])
                + 0.5 * boxsize
            )

        dist = cdist(Fstar["PartType0/Coordinates"][:], [xstar]).flatten()
        closest = dist.argmin()
        hsml = F["PartType0/SmoothingLength"][:][closest]

        if options["--v"] == "comoving":
            vstar = F["PartType0/Velocities"][:][closest]
        else:
            vstar = np.array(
                [
                    float(s) * 1e5 / F["Header"].attrs["UnitVelocity_In_CGS"]
                    for s in options["--v"].split(",")
                ]
            )

        age = (
            float(options["--age"])
            * 3.1541e13
            / (
                F["Header"].attrs["UnitLength_In_CGS"]
                / F["Header"].attrs["UnitVelocity_In_CGS"]
            )
        )

        Fstar["PartType5"].create_dataset(
            "ParticleIDs", data=[F["PartType0/ParticleIDs"][:].max() + 1]
        )
        Fstar["PartType5"].create_dataset("Coordinates", data=[xstar])
        Fstar["PartType5"].create_dataset("Velocities", data=[vstar])
        Fstar["PartType5"].create_dataset("Masses", data=[M])
        Fstar["PartType5"].create_dataset("Potential", data=[0])
        Fstar["PartType5"].create_dataset("ZAMS_Mass", data=[M])
        Fstar["PartType5"].create_dataset("BH_Mass", data=[M])
        Fstar["PartType5"].create_dataset("BH_Mass_AlphaDisk", data=[0])
        Fstar["PartType5"].create_dataset("BH_Mdot", data=[0])
        Fstar["PartType5"].create_dataset("Mass_D", data=[0])
        Fstar["PartType5"].create_dataset("BH_NProgs", data=[0])
        Fstar["PartType5"].create_dataset("ParticleIDGenerationNumber", data=[0])
        Fstar["PartType5"].create_dataset("ParticleChildIDsNumber", data=[0])
        Fstar["PartType5"].create_dataset("BH_Specific_AngMom", data=[[0, 0, 0]])
        Fstar["PartType5"].create_dataset("BH_AccretionLength", data=[hsml])
        Fstar["PartType5"].create_dataset(
            "Metallicity", data=Fstar["PartType0/Metallicity"][:1]
        )
        Fstar["PartType5"].create_dataset(
            "SinkRadius",
            data=[F["Header"].attrs["Fixed_ForceSoftening_Keplerian_Kernel_Extent"][5]],
        )
        Fstar["PartType5"].create_dataset("ProtoStellarStage", data=[5])
        Fstar["PartType5"].create_dataset("ProtoStellarAge", data=[age])
        Fstar["PartType5"].create_dataset(
            "StarLuminosity_Solar", data=luminosity_tout(M)
        )
        Fstar["PartType5"].create_dataset(
            "ProtoStellarRadius_inSolar", data=radius_tout(M)
        )
        Fstar["PartType5"].create_dataset(
            "SinkInitialMass", data=[F["PartType0/Masses"][:].mean()]
        )
        Fstar["PartType5"].create_dataset(
            "StellarFormationTime", data=[Fstar["Header"].attrs["Time"] - age]
        )
        (Fstar["Header"].attrs).modify("NumPart_ThisFile", [Ngas, 0, 0, 0, 0, 1])
        (Fstar["Header"].attrs).modify("NumPart_Total", [Ngas, 0, 0, 0, 0, 1])
        (Fstar["Header"].attrs).modify("Time", 0)
        Fstar.close()
