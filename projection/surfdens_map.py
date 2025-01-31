#!/usr/bin/env python

"""                                                                            
Generates surface density, mass-weighted density, and volume-weighted density maps

Usage: surfdens_map.py <files> ... [options]

Options:                                                                       
   -h --help                   Show this screen.
   --size=<L>                  Image side length in pc (defaults to box size / 5)
   --center=<X,Y>              Center of the image (defaults to box center)
   --res=<N>                   Resolution of the image [default: 256]   
   --output_path               Output path for images (defaults to /surfacedensity directory next to the snapshot)
   --num_jobs=<N>              Number of snapshots to process in parallel [default: 1]
"""

from os import mkdir
from os.path import isdir
import pathlib
import numpy as np
from docopt import docopt
import h5py
from meshoid.grid_deposition import GridSurfaceDensity
from joblib import Parallel, delayed

np.random.seed(42)

options = docopt(__doc__)
RES = int(options["--res"])
if options["--size"]:
    SIZE = float(options["--size"])
else:
    SIZE = None
if options["--center"]:
    CENTER = np.array([float(s) for s in options["--center"].split(",")])
else:
    CENTER = None
if options["--output_path"]:
    if not isdir(options["--output_path"]):
        mkdir(options["--output_path"])
        OUTPATH = options["--output_path"]
else:
    OUTPATH = None
NUM_JOBS = int(options["--num_jobs"])


def make_surfdens_map_from_snapshot(path):
    """Makes a surface density map from a STARFORGE snapshot"""
    with h5py.File(path, "r") as F:
        x = np.float32(F["PartType0/Coordinates"][:])
        m = np.float32(F["PartType0/Masses"][:])
        h = np.float32(F["PartType0/SmoothingLength"][:])
        rho = np.float32(F["PartType0/Density"][:])
        boxsize = F["Header"].attrs["BoxSize"]
    if SIZE:
        size = SIZE
    else:
        size = 0.2 * boxsize
    if CENTER:
        center = CENTER
    else:
        center = 0.5 * np.array(3 * [boxsize])
    dx = size / (RES - 1)
    h = h.clip(dx, np.inf)
    sigmagas = GridSurfaceDensity(m, x, h, center, size, RES, parallel=True)
    rho_mass = GridSurfaceDensity(m * rho, x, h, center, size, RES, parallel=True) / sigmagas
    rho_vol = sigmagas / GridSurfaceDensity(m / rho, x, h, center, size, RES, parallel=True)
    X = np.linspace(dx / 2 - size / 2, size / 2 - dx / 2, RES) + center[0]
    Y = np.linspace(dx / 2 - size / 2, size / 2 - dx / 2, RES) + center[1]
    X, Y = np.meshgrid(X, Y)
    Y = Y[::-1]

    fname = path.split("/")[-1].replace(".hdf5", ".surfacedensity.hdf5")
    if OUTPATH:
        imgpath = OUTPATH + fname
    else:
        outdir = str(pathlib.Path(path).parent.resolve()) + "/surfacedensity/"
        if not isdir(outdir):
            mkdir(outdir)
        imgpath = outdir + fname
    with h5py.File(imgpath, "w") as F:
        F.create_dataset("X_pc", data=X)
        F.create_dataset("Y_pc", data=Y)
        F.create_dataset("SurfaceDensity_Msun_pc2", data=sigmagas)
        F.create_dataset("Density_Mass_Msun_pc3", data=rho_mass)
        F.create_dataset("Density_Vol_Msun_pc3", data=rho_vol)


def main():
    """Runs surface density mapping on input snapshots"""
    Parallel(n_jobs=NUM_JOBS)(delayed(make_surfdens_map_from_snapshot)(f) for f in options["<files>"])


if __name__ == "__main__":
    main()
