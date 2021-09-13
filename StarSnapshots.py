#!/usr/bin/env python
# Creates a copy of snapshots with only PartType5 data, with path <snapshot directory>/stars/snapshot_NNN_stars.hdf5
# usage: StarSnapshot.py file1 file2 file3 file4...
# ACHTUNG ACHTUNG: This runs in parallel. Don't run it on the head node :P

from sys import argv
from time import sleep
import h5py
import numpy as np
import multiprocessing 
from os.path import getsize
from multiprocessing import Pool
from shutil import move
from os import system, mkdir
from os.path import isdir, isfile
nproc = 16 #multiprocessing.cpu_count()

stuff_to_skip = "DivergenceOfMagneticField", "DivBcleaningFunctionPhi", "DivBcleaningFunctionGradPhi", "StarFormationRate", "BH_Dist", "TurbulenceDissipation", "Vorticity", "TurbulenceDriving"
types_to_skip = "PartType0", "PartType3"

def CompressFile(f):
#    print(f, getsize(f)/1e9)
#    sleep(10)
#    return
    stardir = f.split("snapshot_")[0] + "/stars" #sn.replace("snapshot_","/stars")

    if "_stars.hdf5" in f or "_t.hdf5" in f:
        system("rm " + f)
        return
#    N = int(f.split("Res")[1].split("_")[0])**3
    F = h5py.File(f,'r')
#    stardir = "./stars"
#    if not isdir(stardir): mkdir(stardir)
    f2name = stardir +"/" + f.split("/")[-1].replace('.hdf5', '_stars.hdf5')
    if isfile(f2name): return


#    n = np.array(F["PartType0"]["Density"])*29.9
#    cut = (n>1) * (np.array(F["PartType0"]["Masses"]) > 1.1e-4)
#    mach = np.average(np.sum(np.array(F["PartType0"]["Velocities"])[cut]**2,axis=1))**0.5 / 200

    if not "Header" in F.keys(): return
#    f2name = f.replace(".hdf5","_stars.hdf5").replace("snapshot","./stars/snapshot")

    F2 = h5py.File(f2name,'w')
    F.copy("Header", F2)
#    F2.create_dataset("Mach", data=mach)
    for k in F.keys():
        if k=="Header": continue
        if k in types_to_skip: continue
        F2.create_group(k)
        for k2 in F[k].keys():
#            if not k2 in ["Masses", "BH_Mass", "BH_Mdot", "BH_Dust_Mass","IMFFormationProperties"]: continue #k2 != "Masses" and k2 != "BH_Mass": continue
            if k2 in stuff_to_skip: continue
            F2[k].create_dataset(k2, data=F[k][k2], compression="gzip", fletcher32=True, shuffle=True, chunks=True)
    F2.close()
    F.close()
    
if __name__ == "__main__":
    filenames = [f for f in argv[1:]]
    filenames = np.array(filenames)
    Pool(nproc).map(CompressFile, (f for f in filenames))


