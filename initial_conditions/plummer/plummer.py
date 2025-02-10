#!/usr/bin/env python

"""
Generates a GIZMO initial conditions HDF5 file realizing a Plummer model in units where G=1

Usage:
./plummer.py <number of particles> <total mass> <scale radius>
"""

import numpy as np
from scipy.optimize import brentq
import h5py

from sys import argv

G = 1

if len(argv) > 1:
    N = int(float(argv[1]) + 0.5)
else:
    N = 32**3
if len(argv) > 1:
    m = float(argv[2])
else:
    m = 1.0
if len(argv) > 2:
    a = float(argv[3])
else:
    a = 1.0

x = np.arange(0, 1, 1.0 / N) + np.random.rand(N) / N
r = np.sqrt(x ** (2.0 / 3) * (1 + x ** (2.0 / 3) + x ** (4.0 / 3)) / (1 - x**2))
phi = np.random.rand(N) * 2 * np.pi
theta = np.arccos(2.0 * np.random.rand(N) - 1.0)

x = np.c_[r * np.cos(phi) * np.sin(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(theta)]

phi = -((1 + r**2) ** -0.5)
v_e = (-2 * phi) ** 0.5


def cdf(Rp, rr):
    """CDF of velocity distribution"""
    return (
        2
        * (
            Rp * np.sqrt(1 - Rp**2) * (-105 + 1210 * Rp**2 - 2104 * Rp**4 + 1488 * Rp**6 - 384 * Rp**8)
            + 105 * np.arcsin(Rp)
        )
    ) / (105.0 * np.pi) - rr


Qs = np.array([brentq(cdf, 0, 1, args=(R,)) for R in np.random.rand(N)])

v = v_e * Qs

phi_v = np.random.rand(N) * 2 * np.pi
theta_v = np.arccos(2.0 * np.random.rand(N) - 1.0)

v = np.c_[v * np.cos(phi_v) * np.sin(theta_v), v * np.sin(phi_v) * np.sin(theta_v), v * np.cos(theta_v)]
boxsize = 10.0
F = h5py.File("plummer_%d_m%g_a%g.hdf5" % (int(N ** (1.0 / 3) + 0.5), m, a), "w")
F.create_group("PartType1")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [0, len(x), 0, 0, 0, 0]
F["Header"].attrs["MassTable"] = [0, float(m) / N, 0, 0, 0, 0]
F["Header"].attrs["BoxSize"] = boxsize
F["Header"].attrs["Time"] = 0.0
F["PartType1"].create_dataset("Coordinates", data=x * a + boxsize / 2)
F["PartType1"].create_dataset("Potential", data=phi * float(m) / a)
F["PartType1"].create_dataset("Velocities", data=v * (float(m) / a) ** 0.5 * G**0.5)
F["PartType1"].create_dataset("Masses", data=np.repeat(float(m) / N, N))
F["PartType1"].create_dataset("ParticleIDs", data=np.arange(N))
F.close()
