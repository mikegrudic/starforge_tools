#!/usr/bin/env python
import h5py
from sys import argv
from glob import glob
from natsort import natsorted
import numpy as np

for run in argv[1:]: # give the output folders of the runs you want to look at    
    zams_mass_dict = {}
    try:
        snaps = glob(run+"/snapshot_*.hdf5")
        print(snaps)
        for s in snaps:
            print(s)
#            if "stars" in s: continue
            with h5py.File(s,'r') as F:
                if not "PartType5" in F.keys(): continue
                ids = F["PartType5/ParticleIDs"][:]
                mstar = F["PartType5/BH_Mass"][:]
                for i in range(len(ids)):
                    star_id, star_mass = ids[i], mstar[i]
                    if not star_id in zams_mass_dict.keys(): 
                        zams_mass_dict[star_id] = star_mass
                    else: 
                        zams_mass_dict[star_id] = max(star_mass, zams_mass_dict[star_id])                                
    except:
        raise("problem getting IMF for " + run)

    M = float(run.split("/output")[0].split("M")[-1].split("_R")[0])
    R = float(run.split("/output")[0].split("_R")[1].split("_")[0])
    if "_z" in run:
        Z = float(run.split("_z")[1].split("/")[0])
    else:
        Z = 1
    if "_isrf" in run:
        isrf = float(run.split("_isrf")[1].split("/")[0])
    else:
        isrf = 1
    mstar = np.sum([zams_mass_dict[i] for i in zams_mass_dict.keys()])
    if "_B" in run:
        mu = float(run.split("_B")[1].split("_")[0])
        mu = 4.2 * (mu/0.01)**-0.5
    else:
        mu = 4.2
    if "alpha" in run:
        alpha = float(run.split("alpha")[1].split("/")[0])
    else:
        alpha=  2

    seed = 42
    for i in range(1,11):
        if "_%d/"%i in run: seed = i

    header = "#(0) ID (1) Max mass (2) GMC mass (3) GMC radius (4) GMC metallicity (5) ISRF (6) GMC virial parameter (7) GMC mass-to-flux ratio (8) seed (9) total stellar mass formed"

    np.savetxt(run + "/IMF.dat", np.array([[i, zams_mass_dict[i], M, R, Z, isrf, alpha, mu, seed, mstar] for i in zams_mass_dict.keys()]), header=header)
