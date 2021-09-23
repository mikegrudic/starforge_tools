#!/usr/bin/env python

"""                                                                            
Compute power spectra from STARFORGE snapshots. Runs in parallel

Usage: ComputePowerSpectrum.py <files> ... [options]

Options:                                                                       
   -h --help                  Show this screen.
   --boxsize=<L>              Size of the box on which to compute the power spectrum (defaults to 0.2 * simulation box size, -1 uses the full box size)
   --res=<N>                  Size of the grid on which to compute the power spectrum [default: 256]   
"""

from sys import argv
import h5py
from os.path import isdir,abspath
from os import mkdir
from meshoid import Meshoid
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fftn, fftfreq
from scipy.stats import binned_statistic
from docopt import docopt

options = docopt(__doc__)
snapshot_paths = options["<files>"]

def GetPowerSpectrum(grid, res):
    if grid.shape[0] == 3:
        vk = np.array([fftn(V) for V in grid])
        vkSqr = np.sum(np.abs(vk*vk),axis=0)
    else:
        vk = fftn(grid)
        vkSqr = np.abs(vk)**2
    freqs = fftfreq(res)    
    freq3d = np.array(np.meshgrid(freqs,freqs,freqs,indexing='ij'))
    intfreq = np.int_(np.around(freq3d*res)+0.5)
    kSqr = np.sum(np.abs(freq3d)**2,axis=0)
    intkSqr = np.sum(np.abs(intfreq)**2, axis=0)
    intk = intkSqr**0.5

    kbins = np.arange(intk.max())

    power_in_bin = binned_statistic(intk.flatten(), vkSqr.flatten(), bins=kbins,statistic='sum')[0]
    power_spectrum = power_in_bin / np.diff(kbins) # power density in k space
    power_spectrum[power_spectrum == 0] = np.nan
    return kbins[1:], power_spectrum
    

def ComputePowerSpectra(f, options):#for f in argv[1:]:
    powerspec_gridres = int(options["--res"])
    boxsize = options["--boxsize"]
    with h5py.File(f,"r") as F:
        powerspecpath = abspath(f).split("snapshot_")[0] + "/power_spectrum"
        snapnum = f.split("snapshot_")[1].split(".hdf5")[0]
        if not isdir(powerspecpath): mkdir(powerspecpath)
        
        rho = np.array(F["PartType0"]["Density"])
        n = rho*29.9
        if boxsize is not None:
            if float(boxsize) < 0: boxsize = F["Header"].attrs["BoxSize"]
        else:
            boxsize = 0.2 * F["Header"].attrs["BoxSize"]

        print(boxsize)
        center = np.repeat(0.5*F["Header"].attrs["BoxSize"],3)
        
        cut = (n > 0)
        rho = rho[cut]
        x = np.array(F["PartType0"]["Coordinates"])[cut]
        m = np.array(F["PartType0"]["Masses"])[cut]
        v = np.array(F["PartType0"]["Velocities"])[cut]/1e3
        h = np.array(F["PartType0"]["SmoothingLength"])[cut]
        B = np.array(F["PartType0"]["MagneticField"])[cut]*1e4
        
        M = Meshoid(x,m,h)    
        rhogrid = M.DepositToGrid(m,size=boxsize,res=powerspec_gridres,center=center)
        vgrid = np.array([M.InterpToGrid(v[:,i],size=boxsize,res=powerspec_gridres,center=center) for i in range(3)])
        Bgrid = np.array([M.InterpToGrid(B[:,i],size=boxsize,res=powerspec_gridres,center=center) for i in range(3)])

        powerspectra = []
        for grid in vgrid, Bgrid, rhogrid:
            k, powerspec = GetPowerSpectrum(grid, powerspec_gridres)
            powerspectra.append(powerspec)


        np.savetxt(powerspecpath + "/powerspec_" + snapnum + ".dat", np.c_[k, k * 2*np.pi / boxsize, powerspectra[0],powerspectra[1],powerspectra[2]], header="#COLUMNS: (0) Integer wavenumber k (1) Physical wavenumber (pc^-1) (2) velocity power spectrum (3) magnetic field power spectrum (4) density power spectrum")
        
from multiprocessing import Pool
#[ComputePowerSpectra(f,options) for f in snapshot_paths]
def func(x):
    return ComputePowerSpectra(x,options)
Pool(14).map(func,snapshot_paths)
