#!/usr/bin/env python
"""
Usage: ComputeNeff_nH.py [options]

Options:
    -h --help                Show this screen
    --simPath=<simPath>      Path to simulation directory
    --outPath=<outPath>      Path to output directory [default: ./]
    --snapLow=<snapLow>      Lowest simulation snapshot number
    --snapHigh=<snapHigh>    Highest simulation snapshot number
    --nsnap=<nsnap>          Number of snapshots to process
    --NSIDE=<NSIDE>          Healpix NSIDE parameter (set to 0 for 6-ray approximation) [default: 0]
    --ntasks=<ntasks>        Run in parallel [default: 1]
    --nthread=<nthread>      Number of threads to use. If greater than the number of snaps, defaults to nsnap [default: 1]
"""
import numpy as np
import h5py
import healpy as hp

from meshoid import Meshoid
from pytreegrav import ColumnDensity
import numba

from glob import glob

from docopt import docopt

import matplotlib.pyplot as plt
import matplotlib.colors as colors

mh = 1.67e-24
AV_fac = 6.289e-22


def computeNeff_nH(
    snapi,
    simPath,
    outPath,
    healPix=False,
    NPIX=6,
    healVect=None,
):
    outAvnh = outPath + "avnh_" + str(snapi) + ".out"
    outPlot = outPath + "avnh_" + str(snapi) + ".pdf"
    fileName = simPath + "snapshot_%03d.hdf5" % snapi
    F = h5py.File(fileName)
    UNIT_LENGTH = F["Header"].attrs["UnitLength_In_CGS"]
    UNIT_MASS = F["Header"].attrs["UnitMass_In_CGS"]

    rho = F["PartType0"]["Density"][:] * 6.77e-23  # UNIT_MASS / UNIT_LENGTH**3
    nRho = rho / (1.4 * mh)
    nRho = nRho
    pdata = {}
    for field in "Masses", "Coordinates", "SmoothingLength", "Velocities":
        pdata[field] = F["PartType0"][field][:][:]
    pos, mass, hsml, v = pdata["Coordinates"], pdata["Masses"], pdata["SmoothingLength"], pdata["Velocities"]

    center = np.median(pos, axis=0)
    pos -= center
    rad = np.linalg.norm(pos, axis=1)
    maxRad = 0.9 * np.amax(rad)

    if healPix:
        NCOL = ColumnDensity(pos * UNIT_LENGTH, mass * UNIT_MASS, hsml * UNIT_LENGTH, parallel=True, rays=healVect)
    else:
        NCOL = ColumnDensity(pos * UNIT_LENGTH, mass * UNIT_MASS, hsml * UNIT_LENGTH, parallel=True)
    NCOL /= 1.4 * 1.67e-24  # convert to 1/cm^2
    NCOL += 1e16

    Avhp = NCOL * AV_fac

    eAvhp = np.exp(-Avhp)
    AvEff = -np.log((1 / float(NPIX)) * np.sum(eAvhp, axis=1))
    NEff = AvEff / AV_fac

    np.save(outAvnh, np.array([nRho, NEff]))

    M = Meshoid(pos, mass, hsml)
    res = 2**8
    X = Y = np.linspace(-maxRad, maxRad, res)

    n_space = np.logspace(0, 8, 32)
    Neff_nH = 0.05 * n_space**0.5 / AV_fac

    fig, (ax, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    ax.hexbin(np.log10(nRho), np.log10(NEff), gridsize=64, bins="log", cmap="inferno")
    ax.plot(np.log10(n_space), np.log10(Neff_nH), "k-", lw=2.5, label=r"N $\approx n^{1/2}$")
    ax.set_xlabel(r"$\log n$", fontsize=16)
    ax.set_ylabel(r"$\log N_{\rm eff}({\rm H})$", fontsize=16)
    ax.legend(loc="upper left")

    col_slice = M.Slice(AvEff, center=np.array([0, 0, 0]), size=3, res=res)
    p = ax2.pcolormesh(X, Y, col_slice, norm=colors.LogNorm(vmin=1e20, vmax=1e23), cmap="magma", rasterized=True)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (pc)")
    ax2.set_ylabel("Y (pc)")

    plt.tight_layout()
    fig.savefig(outPlot, bbox_inches="tight")
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":
    args = docopt(__doc__)

    simPath = args["--simPath"]
    if simPath[-1] != "/":
        simPath += "/"

    outPath = args["--outPath"]
    if outPath[-1] != "/":
        outPath += "/"

    snapLow = int(args["--snapLow"])
    snapHigh = int(args["--snapHigh"])
    if snapLow > snapHigh:
        print("Oops! You seemed to have set snapLow > snapHigh. I'll fix that for you.")
        snapLow, snapHigh = snapHigh, snapLow
    nsnap = int(args["--nsnap"])

    NSIDE = int(args["--NSIDE"])
    healPix = False
    if NSIDE > 0:
        healPix = True
        NPIX = hp.nside2npix(NSIDE)
        healVect = np.array(hp.pix2vec(NSIDE, np.arange(NPIX))).T
    else:
        NPIX = 6
        healVect = None

    if nsnap == 1 or snapLow == snapHigh:
        computeNeff_nH(snapLow, simPath, outPath, healPix=healPix, NPIX=NPIX, healVect=healVect)
    else:
        fileList = np.array(glob(simPath + "snapshot_*.hdf5"))
        snapIndices = np.array([int(file.split("_")[-1].split(".")[0]) for file in fileList])

        fileList = fileList[np.argsort(snapIndices)]
        snapIndices = snapIndices[np.argsort(snapIndices)]

        indLow = np.where(snapIndices >= snapLow)[0][0]
        indHigh = np.where(snapIndices <= snapHigh)[0][-1]

        dN = (indHigh + 1 - indLow) // nsnap
        indList = np.linspace(indLow, indHigh, nsnap, endpoint=True).astype(int)
        snapList = snapIndices[indList]

        fileList2 = fileList[indList]

        ntasks = int(args["--ntasks"])
        if ntasks > 1:
            from multiprocessing import Pool
            from functools import partial

            parallel = True
            nthread = int(args["--nthread"])
            if ntasks > nsnap:
                ntasks = nsnap
                print("Defaulted to %d threads" % nsnap)
            if nthread < ntasks:
                print("Too few threads! Defaulting to ntasks = nthread")
                ntasks = nthread
            nthread_per_task = nthread // ntasks
            numba.set_num_threads(nthread_per_task)

        if ntasks > 1:
            with Pool(processes=ntasks) as pool:
                results = pool.map(
                    partial(
                        computeNeff_nH, simPath=simPath, outPath=outPath, healPix=healPix, NPIX=NPIX, healVect=healVect
                    ),
                    snapList,
                )
                pool.close()
                pool.join()
        else:
            for snapi in snapList:
                computeNeff_nH(snapi, simPath, outPath, healPix=healPix, NPIX=NPIX, healVect=healVect)
