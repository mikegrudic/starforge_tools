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

    np.savetxt("IMF.dat", np.array([[i, zams_mass_dict[i]] for i in zams_mass_dict.keys()]))
