#!/usr/bin/env python
# usage: AddDustToSnapshot.py my_snapshot1.hdf5 my_snapshot2.hdf5 ....  will output my_snapshot1_dusty.hdf5, my_snapshot2_dusty.hdf5...
from sys import argv
import h5py
from os.path import isfile
from os import system
import numpy as np

mu = 0.01 # mass-weighted dust to gas ratio

for f in argv[1:]:
    print("Adding dust to " + f)
    with h5py.File(f,'r') as F:
        fname = f.replace(".hdf5","_dusty.hdf5")
        if isfile(fname): system("rm " + fname)
        Fdust = h5py.File(fname,'w')
        for k in F.keys():
            F.copy(k, Fdust)
        Fdust["PartType0"]["Masses"].write_direct((1-mu)*np.array(Fdust["PartType0"]["Masses"]))
        Fdust.create_group("PartType3")
        Ngas = len(np.array(Fdust["PartType0"]["Coordinates"]))
        print(Ngas)
        Fdust["PartType3"].create_dataset("Coordinates", data=np.array(Fdust["PartType0"]["Coordinates"]) * (1 + 1e-6*np.random.normal(size=(Ngas,3))))
        Fdust["PartType3"].create_dataset("Velocities", data=np.array(Fdust["PartType0"]["Velocities"]))
        Fdust["PartType3"].create_dataset("Masses", data=mu*np.array(Fdust["PartType0"]["Masses"]))            
        (Fdust["Header"].attrs).modify("NumPart_ThisFile", [Ngas, 0, 0, Ngas, 0, 0])
        (Fdust["Header"].attrs).modify("NumPart_Total", [Ngas, 0, 0, Ngas, 0, 0])
        Fdust.close()
        
