#!/usr/bin/python
import h5py
from glob import glob
from sys import argv
from natsort import natsorted
import numpy as np

mmin = 0.1
mmax = 100

for run in argv[1:]:
    mdict = {}
    files = sorted(glob(run + "/snapshot*.hdf5"))  # + glob(run + "/stars_only/*.hdf5"))
    for f in files:
        print(f)
        with h5py.File(f, "r") as F:
            if not "PartType5" in F.keys():
                continue
            t = F["Header"].attrs["Time"]
            ids = F["PartType5/ParticleIDs"][:]
            mstar = F["PartType5/BH_Mass"][:]
            msink = F["PartType5/Masses"][:]
            for i, m, ms in zip(ids, mstar, msink):
                if i in mdict.keys():
                    mdict[i].append([t, m, ms])
                else:
                    mdict[i] = [[t, m, ms]]

    F_acc = h5py.File(run + "/accretion_histories.hdf5", "w")
    for i in sorted(mdict.keys()):
        mdict[i] = np.array(mdict[i])
        mdict[i] = mdict[i][mdict[i][:, 0].argsort()]
        t, m, ms = mdict[i].T
        m_ZAMS = m.max()
        if m_ZAMS < mmin or m_ZAMS > mmax:
            continue
        groupname = "Star%d" % i
        F_acc.create_group(groupname)
        F_acc.create_dataset(groupname + "/ZAMS_Mass", data=m_ZAMS)
        F_acc.create_dataset(groupname + "/Time_Myr", data=t * 978.5)
        F_acc.create_dataset(groupname + "/Mass_Msun", data=m)
        F_acc.create_dataset(groupname + "/Mass_Sink_Msun", data=ms)
#    print(mdict)
#    Facc = h5py.File(run+"/accretion_histories.hdf5","w")
