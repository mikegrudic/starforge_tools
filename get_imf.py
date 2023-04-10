#!/usr/bin/env python
import h5py
from sys import argv
from glob import glob
from natsort import natsorted
import numpy as np
from simple_powerlaw_fit import simple_powerlaw_fit

for run in argv[1:]: # give the output folders of the runs you want to look at    
    zams_mass_dict = {}
    t0_dict = {}
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
                t = F["Header"].attrs["Time"]
                for i in range(len(ids)):
                    star_id, star_mass = ids[i], mstar[i]
                    if not star_id in zams_mass_dict.keys(): 
                        zams_mass_dict[star_id] = star_mass
                        t0_dict[star_id] = t
                    else: 
                        zams_mass_dict[star_id] = max(star_mass, zams_mass_dict[star_id])                                
                        t0_dict[star_id] = min(t, t0_dict[star_id])
    except:
        continue
       # raise ValueError("problem getting IMF for " + run)

    M = float(run.split("/output")[0].split("M")[-1].split("_R")[0])
    R = float(run.split("/output")[0].split("_R")[-2].split("_")[0])
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

    header = "(0) ID (1) Max mass (2) seed formation time (3) GMC mass (4) GMC radius (5) GMC metallicity (6) ISRF (7) GMC virial parameter (8) GMC mass-to-flux ratio (9) seed (10) total stellar mass formed\n"
    masses = np.array([zams_mass_dict[i] for i in zams_mass_dict.keys()])
    header += "SFE: %g\n"%(masses.sum()/M)
    if len(masses)>10:
        alpha_lower, alpha_med, alpha_upper = simple_powerlaw_fit(masses,xmin=1,xmax=10)
        header += "IMF params: %g %g %g %g\n"%(alpha_lower,alpha_med,alpha_upper,masses.max())

    np.savetxt(run + "/IMF.dat", np.array([[i, zams_mass_dict[i], t0_dict[i], M, R, Z, isrf, alpha, mu, seed, mstar] for i in zams_mass_dict.keys()]), header=header)


