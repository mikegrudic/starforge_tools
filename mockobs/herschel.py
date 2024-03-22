#!/usr/bin/env python

"""                                                                            
Generates far-IR images of simulation snapshots that have dust temperature
information.

Usage: herschel.py <files> ... [options]

Options:                                                                       
   -h --help                   Show this screen.
   --size=<L>                  Image side length in pc (defaults to box size / 5)
   --center=<X,Y>              Center of the image (defaults to box center)
   --res=<N>                   Resolution of the image [default: 1024]   
   --wavelengths=<l1,l2,etc>   Wavelengths in micron to image [default: 150, 250, 350, 500]
   --output_path               Output path for images (defaults to cwd)
   --num_jobs=<N>              Number of snapshots to process in parallel [default: 1]
"""

from os import mkdir
from os.path import isdir
from astropy import constants
import astropy.units as u
import pathlib
import numpy as np
from docopt import docopt
import h5py
from scipy.optimize import curve_fit
from meshoid.radiation import dust_emission_map
from joblib import Parallel, delayed

options = docopt(__doc__)
WAVELENGTHS = np.array([float(s) for s in options["--wavelengths"].split(",")])
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


def make_herschel_map_from_snapshot(path):
    """Makes a dust emission map from a STARFORGE snapshot"""
    with h5py.File(path, "r") as F:
        x = np.float32(F["PartType0/Coordinates"][:])
        m = np.float32(F["PartType0/Masses"][:])
        h = np.float32(F["PartType0/SmoothingLength"][:])
        Tdust = np.float32(F["PartType0/Dust_Temperature"][:])
        boxsize = F["Header"].attrs["BoxSize"]
    if SIZE:
        size = SIZE
    else:
        size = 0.2 * boxsize
    if CENTER:
        center = CENTER
    else:
        center = 0.5 * np.array(3 * [boxsize])
    intensity = dust_emission_map(x, m, h, Tdust, size, RES, WAVELENGTHS, center)

    dx = size / (RES - 1)
    X = np.linspace(dx / 2, size - dx / 2, RES) + center[0]
    Y = np.linspace(dx / 2, size - dx / 2, RES) + center[1]
    X, Y = np.meshgrid(X, Y)
    Y = Y[::-1]

    fname = path.split("/")[-1].replace(".hdf5", ".herschel.hdf5")
    if OUTPATH:
        imgpath = OUTPATH + fname
    else:
        outdir = str(pathlib.Path(path).parent.resolve()) + "/herschel/"
        if not isdir(outdir):
            mkdir(outdir)
        imgpath = outdir + fname
    with h5py.File(imgpath, "w") as F:
        F.create_dataset("X_pc", data=X)
        F.create_dataset("Y_pc", data=Y)
        F.create_dataset("Intensity_cgs", data=intensity)


def main():
    # print(f"jobs={NUM_JOBS}")
    Parallel(n_jobs=NUM_JOBS)(
        delayed(make_herschel_map_from_snapshot)(f) for f in options["<files>"]
    )


if __name__ == "__main__":
    main()
