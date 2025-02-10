#!/usr/bin/env python
"""
Generates a GIZMO HDF5 initial conditions file consisting a binary system of unit mass and semimajor axis
and arbitrary eccentricity and mass ratio, in units where G=1

Usage:
./binary.py <eccentricity> <mass ratio>
"""
import numpy as np
import h5py
from sys import argv

ecc = float(argv[1])
mass_ratio = float(argv[2])
m1 = 1 / (1 + mass_ratio)
m2 = 1 - m1

x = np.array([[-m2, 0, 0], [m1, 0, 0]])
m = np.array([m1, m2])
v = (1 - ecc) ** 0.5 * np.array([[0.0, m2, 0], [0, -m1, 0]])
boxsize = 10.0
F = h5py.File("e%g.hdf5" % ecc, "w")
F.create_group("PartType5")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, len(x)]
F["Header"].attrs["BoxSize"] = boxsize
F["Header"].attrs["Time"] = 0.0
F["PartType5"].create_dataset("Coordinates", data=x + boxsize / 2)
F["PartType5"].create_dataset("Velocities", data=v)
F["PartType5"].create_dataset("Masses", data=m)  # np.repeat(1./len(m),2))
F["PartType5"].create_dataset("ParticleIDs", data=[1, 2])
F.close()
