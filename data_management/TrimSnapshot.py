#!/usr/bin/env python

# usage: TrimSnapshot.py file1 file2 file3 file4...
# ACHTUNG: Overwriting will irreversibly alter the snapshot files you run it on! Be absolutely sure you wanna do this.
# ACHTUNG ACHTUNG: This uses every available thread. Don't run it on the head node :P

from sys import argv
from time import sleep
import h5py
import numpy as np
import multiprocessing
from os.path import getsize, islink
from multiprocessing import Pool
from shutil import move
from os import system, walk

nproc = int(argv[1])  # multiprocessing.cpu_count()

stuff_to_skip = (
    "StarFormationRate",
    "BH_Dist",
    "TurbulenceDissipation",
    "Vorticity",
    "VelocityDivergence",
    "DivergenceOfMagneticField",
    "DivBcleaningFunctionPhi",
    "DivBcleaningFunctionGradPhi",
)
stuff_to_store_at_half_precision = []
overwrite = True
types_to_skip = []  # add e.g. "PartType0" to this list if you want to DELETE that particle type (careful with this!)


def is_this_already_compressed(f):
    print(f)
    if islink(f):
        return True
    if "_bug" in f:
        return True
    if not "hdf5" in f:
        return True

    try:
        with h5py.File(f, "r") as F:
            if F["PartType0/Masses"].compression == "gzip":
                print(f"{f} is already compressed.")
                return True
            else:
                return False
    except:
        print(f)
        return True


def CompressFile(f):
    if islink(f):
        return
    if "_bug" in f:
        return
    if not "hdf5" in f:
        return
    print(f, getsize(f) / 1e9)
    #    if getsize(f)/1e9 < 5: return
    #    if getsize(f)/1e9 > 10: return
    if "_t.hdf5" in f:
        system("rm " + f)
        return
    F = h5py.File(f, "r")
    if not "Header" in F.keys():
        return
    f2name = f.replace(".hdf5", "_t.hdf5")
    F2 = h5py.File(f2name, "w")
    F.copy("Header", F2)
    for k in F.keys():
        if k == "Header":
            continue
        if k in types_to_skip:
            continue
        F2.create_group(k)
        for k2 in F[k].keys():
            if k2 in stuff_to_skip:
                continue
            if k2 in stuff_to_store_at_half_precision:
                F2[k].create_dataset(
                    k2,
                    data=np.float16(np.log10(F[k][k2])),
                    compression="gzip",
                    fletcher32=True,
                    shuffle=True,
                    chunks=True,
                )
            else:
                F2[k].create_dataset(k2, data=F[k][k2], compression="gzip", fletcher32=True, shuffle=True, chunks=True)

    F2.close()
    F.close()
    if overwrite:
        move(f2name, f)


def main():
    files_to_compress = []
    for root, dirs, files in walk("."):
        if "stars_" in root:
            continue
        for f in files:
            if ".hdf5" in f and "snapshot" in f:
                if "_stars" in f:
                    continue
                path = root + "/" + f
                if not is_this_already_compressed(path):
                    files_to_compress.append(path)

    #    print(files_to_compress)

    # filenames = [f for f in argv[2:]]
    sizes = np.array([getsize(f) for f in files_to_compress])
    print(np.c_[files_to_compress, sizes][sizes.argsort()[::-1]])
    #    filenames = np.array(files_to_compress)[sizes.argsort()]#[::-1]
    print(np.c_[files_to_compress, sizes])
    if nproc > 1:
        Pool(nproc).map(CompressFile, (f for f in files_to_compress), chunksize=1)
    else:
        [CompressFile(f) for f in files_to_compress]


if __name__ == "__main__":
    main()
