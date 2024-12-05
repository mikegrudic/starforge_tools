#!/usr/bin/env python

"""
Given the output directory of a run, sets up a subdirectory from which a dry
N-body continuation of a STARFORGE run can be run.
"""

import os
from os import mkdir, system
from os.path import isdir, abspath
from shutil import copyfile
from sys import argv
from natsort import natsorted
from glob import glob
import h5py
import numpy as np

for path in argv[1:]:
    rundir = abspath(path)
    if not "output" in rundir:
        continue
        
    nbody_path = rundir+"/nbody_evolution"
    if not isdir(nbody_path):
        mkdir(nbody_path)

    copyfile(os.environ["HOME"]+"/gizmo_nbody/GIZMO", nbody_path+"/GIZMO")
    system("chmod +x " + nbody_path+'/GIZMO')
    copyfile(os.environ["HOME"]+"/gizmo_nbody/run_nbody.sh", nbody_path+"/run_nbody.sh")
    copyfile(rundir.split("/output")[0] +"/params.txt", nbody_path+"/params.txt")
    
    system("sed -ie 's/BlackHoleMaxAccretionRadius/BlackHoleMaxAccretionRadius  1e-7 %/g' " + nbody_path + "/params.txt")

    last_snapshot = natsorted(glob(rundir+"/snapshot*.hdf5"))[-1]
    F_orig = h5py.File(last_snapshot,'r')
    nbody_IC = last_snapshot.replace("/snapshot","/nbody_evolution/snapshot")
    Fdry = h5py.File(nbody_IC,'w')
    for k in F_orig.keys():
        if k != "PartType0":
            F_orig.copy(k, Fdry)

    F_orig.close()

    Nstar = len(np.array(Fdry["PartType5"]["Coordinates"]))
    (Fdry["Header"].attrs).modify("NumPart_ThisFile", [0, 0, 0, 0, 0, Nstar])
    (Fdry["Header"].attrs).modify("NumPart_Total", [0, 0, 0, 0, 0, Nstar])
    Fdry.close()
    
    last_snapshot = last_snapshot.split("/")[-1].replace(".hdf5","")
    system("sed -ie 's/InitCondFile/InitCondFile " + last_snapshot + " %/g' " + nbody_path + "/params.txt")
