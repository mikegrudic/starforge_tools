#!/usr/bin/env python

"""                                                                            
Generates far-IR images of simulation snapshots that have dust temperature
information.

Usage: dustemission.py <files> ... [options]

Options:                                                                       
   -h --help                   Show this screen.
   --size=<L>                  Image side length in pc (defaults to box size / 5)
   --center=<X,Y>              Center of the image (defaults to box center)
   --res=<N>                   Resolution of the image [default: 1024]   
   --wavelengths=<l1,l2,etc>   Wavelengths in micron to image [default: 150, 250, 350, 500]
   --output_path               Output path for images (defaults to /dustemission 
                               directory next to the snapshot)   
   --num_jobs=<N>              Number of snapshots to process in parallel [default: 1]          
"""

from os import mkdir
from os.path import isdir
import pathlib
import numpy as np
from docopt import docopt
import h5py
from meshoid.radiation import (
    dust_emission_map,
    modified_blackbody_fit_image,
    modified_blackbody_fit_gaussnewton,
)
from meshoid.grid_deposition import GridSurfaceDensity
from joblib import Parallel, delayed

np.random.seed(42)

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

SOLAR_Z = 0.0142


def make_dustemission_map_from_snapshot(path):
    """Makes a dust emission map from a STARFORGE snapshot"""
    with h5py.File(path, "r") as F:
        x = np.float32(F["PartType0/Coordinates"][:])
        m = np.float32(F["PartType0/Masses"][:])
        h = np.float32(F["PartType0/SmoothingLength"][:])
        Z = F["PartType0/Metallicity"][:]
        if len(Z.shape) == 2:
            Z = Z[:, 0] / SOLAR_Z
        else:
            Z = Z / SOLAR_Z

        if "PartType0/Dust_Temperature" in F.keys():
            Tdust = np.float32(F["PartType0/Dust_Temperature"][:])
        else:
            Tdust = np.repeat(20, len(x))
        Tdust_avg = np.average(Tdust, weights=m)

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
    intensity = dust_emission_map(x, m * Z, h, Tdust, size, RES, WAVELENGTHS, center)

    # add noise
    # SNR = 60
    # noise_norm = intensity/SNR + 1e-18 * 10**(-np.log10(WAVELENGTHS[None,None,:]/500) * 2)
    # SNR_tot = noise_norm / intensity
    # N_eff = SNR_tot**-2
    # noise = np.random.poisson(N_eff)/N_eff * intensity - intensity
    h = h.clip(dx, 1e100)
    sigmagas = GridSurfaceDensity(m, x, h, center, size, RES)
    X = np.linspace(0.5 * (dx - size), 0.5 * (size - dx), RES) + center[0]
    Y = np.linspace(0.5 * (dx - size), 0.5 * (size - dx), RES) + center[1]
    X, Y = np.meshgrid(X, Y, indexing="ij")

    fname = path.split("/")[-1].replace(".hdf5", ".dustemission.hdf5")
    if OUTPATH:
        imgpath = OUTPATH + fname
    else:
        outdir = str(pathlib.Path(path).parent.resolve()) + "/dustemission/"
    if not isdir(outdir):
        mkdir(outdir)
    imgpath = outdir + fname
    with h5py.File(imgpath, "w") as F:
        F.create_dataset("Wavelengths_um", data=WAVELENGTHS)
        F.create_dataset("X_pc", data=X)
        F.create_dataset("Y_pc", data=Y)
        F.create_dataset("Intensity_cgs", data=intensity)
        #        F.create_dataset("Intensity_noise_cgs", data=noise)
        F.create_dataset("SurfaceDensity_Msun_pc2", data=sigmagas)

        if len(WAVELENGTHS) >= 3:
            # fit each pixel to a modified blackbody as an observer would
            params = modified_blackbody_fit_image(intensity, WAVELENGTHS)
            tau0, beta, Tdust_fit = (
                params[:, :, 0],
                params[:, :, 1],
                params[:, :, 2],
            )

            F.create_dataset("Fit_Tau500um", data=tau0)
            F.create_dataset("Fit_Beta", data=beta)
            F.create_dataset("Fit_Tdust", data=Tdust_fit)
            F.create_dataset("Tdust_massweighted", data=Tdust_avg)

            params_avg = modified_blackbody_fit_gaussnewton(
                intensity.mean((0, 1)), WAVELENGTHS
            )

            tau0, beta, Tdust_fit = params_avg

            F.create_dataset("SEDFit_Tau500um", data=tau0)
            F.create_dataset("SEDFit_Beta", data=beta)
            F.create_dataset("SEDFit_Tdust", data=Tdust_fit)


def main():
    """Runs dust mapping and SED fitting on input snapshots"""
    Parallel(n_jobs=NUM_JOBS)(
        delayed(make_dustemission_map_from_snapshot)(f) for f in options["<files>"]
    )


if __name__ == "__main__":
    main()
