from os import walk, system, chdir, getcwd
from os.path import isfile, isdir
import numpy as np
from sys import argv
from glob import glob
from natsort import natsorted

source_codes = []
runs = []

count = 0

def main():
    ORIG_CWD = getcwd()
    for root, dirs, files in walk(argv[1]):
        for d in dirs:
            if "output_stirring" in d:
                system(f"mv {root+'/'+d} {root+'/'+d.replace('output_stirring', 'output_turbsphere_stirring')}")
            if "output_driving" in d:
                system(f"mv {root+'/'+d} {root+'/'+d.replace('output_driving', 'output_turbsphere_driving')}")
            if "output_nodriving" in d:
                system(f"mv {root+'/'+d} {root+'/'+d.replace('output_nodriving', 'output_turbsphere_nodriving')}")
        
        if "M" in root and sum(["output" in d for d in dirs]): # we have a simulation directory                
            print("Simulation:", root,dirs)
            if len(glob(root+"/*/RUNNING")) == 0 and isdir(root+"/spcool_tables"): 
                print("rm -rf " + root + "/spcool_tables")
                system("rm -rf " + root + "/spcool_tables")
                
            runs.append(root)

        if "blackhole_details" in root: # blackhole files 
            if isfile(root.split("blackhole_details")[0] + "/RUNNING"): continue
            bhswallow_path = root.split("blackhole_details")[0] + "/bhswallow.dat"
            bhformation_path = root.split("blackhole_details")[0] + "/bhformation.dat"
            with open(bhswallow_path,"w") as F:
                F.write(bhswallow_header)
            with open(bhformation_path,"w") as F:
                F.write(bhformation_header)
            system("cat " + root + "/bhswallow*.txt >> " + bhswallow_path)
            system("cat " + root + "/bhformation*.txt >> " + bhformation_path)
            tarpath = root + ".tar.gz"
            system("tar -czf " + tarpath + " " + root)
            if isfile(tarpath): 
                print("rm -rf " + root)
                system("rm -rf " + root)
            
    #        system("cat "+ root + "/bhswallow_* > bhswallow.txt")
        elif ".git" in dirs and "Makefile" in files and "run.c" in files: # we have a source code directory
            print("Source code:", root,dirs)
    #        tarpath = root + ".tar.gz"
    #        print(root, tarpath)
    #        system("tar -czvf " + tarpath + " " +  root)
    #        if isfile(tarpath): system("rm -rf " + root)
            source_codes.append(root)
        
        





bhformation_header = """
(0) Time
(1) ID
(2) Mass
(3) X
(4) Y
(5) Z
(6) vx
(7) vy
(8) vz
(9) Bx
(10) By
(11) Bz
(12) u
(13) rho
(14) cs
(15) cell size
(16) column density
(17) velocity gradient^2
(18) min dist to nearest star
"""
bhswallow_header = """
# (0) Time
# (1) ID
# (2) Mass
# (3) X
# (4) Y
# (5) Z
# (6) Accreta ID
# (7) Accreta mass
# (8) dx
# (9) dy
# (10) dz
# (11) dvx
# (12) dvy
# (13) dvz
# (14) Accreta specific internal energy
# (15) Bx
# (16) By
# (17) Bz
# (18) Density
"""


if __name__=="__main__":
    main()

    print(runs)
    with open("runs.txt","w") as F:
        for r in runs:
            F.write(r + "\n")

    with open("source.txt","w") as F:
        for r in source_codes:
            F.write(r+ "\n")
