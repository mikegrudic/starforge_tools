#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
#from scipy.optimize import brentq
import h5py
from sys import argv

e = argv[1]

x = np.array([[-0.1,0,0], [0.9,0,0]])
m = np.array([0.9,0.1])
v = (1 - e)**0.5 * np.array([[0.,0.1,0], [0,-0.9,0]])#  - 2*x
print(x)

#v = np.c_[v*np.cos(phi_v)*np.sin(theta_v), v*np.sin(phi_v)*np.sin(theta_v), v*np.cos(theta_v)]
boxsize = 10.
F = h5py.File("e%g.hdf5"%e)
F.create_group("PartType5")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [0,0,0,0,0,len(x)]
#F["Header"].attrs["MassTable"] = [0,0,0,0,0,1./len(m)]
F["Header"].attrs["BoxSize"] = boxsize
F["Header"].attrs["Time"] = 0.0
F["PartType5"].create_dataset("Coordinates", data=x + boxsize/2)
F["PartType5"].create_dataset("Velocities", data=v)
F["PartType5"].create_dataset("Masses", data=m)#np.repeat(1./len(m),2))
F["PartType5"].create_dataset("ParticleIDs", data=[1,2])
F.close()
