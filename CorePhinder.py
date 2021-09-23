#!/usr/bin/env python
"""                                                                            
Catalogues self-gravitating iso-density contours surrounding density peaks whose enclosed gas has a specified virial parameter, and does not contain any smaller self-gravitating substructures (i.e. first-crossing objects).

Usage: CloudPhinder.py <files> ... [options]

Options:                                                                       
   -h --help                  Show this screen.
   --recompute_potential      Whether to trust the potential found in the snapshot, and if not, recompute it (only needed for potential mode) [default: False]
   --G=<G>                    Gravitational constant to use; should be consistent with what was used in the simulation. [default: 4.3007e3]
   --boxsize=<L>              Box size of the simulation; for neighbour-search purposes. [default: None]
   --cluster_ngb=<N>          Length of particle's neighbour list. [default: 32]
   --nmin=<n>                 Minimum particle number density to cut at, in cm^-3 [default: 10]
   --alpha_crit=<f>           Critical virial parameter to be considered bound [default: 1.0]
   --ignore_B                 Ignore magnetic field
   --np=<N>                   Number of cores to run on [default: 1]
   --ngrav=<N>                Number of cores per python process to parallelize gravity [default: 4]
"""

#alpha_crit = 1000
ntree = 1000
pytreegrav_parallel = True

import h5py
from numba import jit, vectorize
from scipy.spatial import cKDTree
from time import time as time_seconds
from numba.typed import List
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import numpy as np
import pytreegrav
from pytreegrav import Potential, Octree, PotentialWalk, Potential_bruteforce
from sys import argv
from docopt import docopt
from collections import OrderedDict
from os import path, mkdir
from joblib import Parallel, delayed
from numba import set_num_threads

max_num_crossings = 1

def SaveArrayDict(path, arrdict):
    """Takes a dictionary of numpy arrays with names as the keys and saves them in an ASCII file with a descriptive header"""
    header = ""
    offset = 0
    
    for i, k in enumerate(arrdict.keys()):
        if type(arrdict[k])==list: arrdict[k] = np.array(arrdict[k])
        if len(arrdict[k].shape) == 1:
            header += "(%d) "%offset + k + "\n"
            offset += 1
        else:
            header += "(%d-%d) "%(offset, offset+arrdict[k].shape[1]-1) + k + "\n"
            offset += arrdict[k].shape[1]
            
    data = np.column_stack([b for b in arrdict.values()])
    data = data[(-data[:,0]).argsort()] 
    np.savetxt(path, data, header=header,  fmt='%.15g', delimiter='\t')

######## Energy functions ######## 
def KE(c, x, m, h, v, u):
    """ c - clump index
        x - 2D array of positions (Nclump,N_this_clump)
        m - 2D array of masses (Nclump,N_this_clump)
        h - 2D array of smoothing lengths (Nclump,N_this_clump)
        v - 2D array of velocities (Nclump,N_this_clump)
        u - 2D array of internal energies (Nclump,N_this_clump)
    """
    xc, mc, vc, hc = x[c], m[c], v[c], h[c]

    ## velocity w.r.t. com velocity of clump
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return (mc*(vSqr/2 + u[c])).sum()

def PE(c, x, m, h, v, u):
    phic = pytreegrav.Potential(x[c], m[c], h[c], G=4.3007e3, theta=0.7,parallel=pytreegrav_parallel)
    return 0.5*(phic*m[c]).sum()

def InteractionEnergy(
    x, m, h,
    group_a, tree_a,
    particles_not_in_tree_a,
    group_b, tree_b,
    particles_not_in_tree_b):

    xb, mb, hb = x[group_b], m[group_b], h[group_b]    
    if tree_a:
        ## evaluate potential from the particles in the tree
        phi = pytreegrav.PotentialTarget(
            xb,
            None, ## pos source
            None, ## mass source
            tree=tree_a, ## source tree
            G=4.3007e3,
            theta=1,
            parallel=pytreegrav_parallel)

        ## brute force the particles not in the tree
        if len(particles_not_in_tree_a):
            arr = particles_not_in_tree_a
            xa = arr[:,:3] #np.take(x, particles_not_in_tree_a,axis=0)
            ma = arr[:,3] #np.take(m, particles_not_in_tree_a)
            ha = arr[:,4] #np.take(h, particles_not_in_tree_a)
            phi += pytreegrav.PotentialTarget(xb, xa, ma, h_target=hb, h_source=ha,G=4.3007e3,parallel=pytreegrav_parallel)
    else:
        ## have to brute force all the particles
        xa, ma, ha = x[group_a], m[group_a], h[group_a]        
        phi = pytreegrav.PotentialTarget(xb, xa, ma, h_source=ha, h_target=hb, G=4.3007e3,parallel=pytreegrav_parallel)
    potential_energy = (mb*phi).sum()
    return potential_energy
    
def VirialParameter(c, x, m, h, v, u):
    ke, pe = KE(c,x,m,h,v,u), PE(c,x,m,h,v,u)
    return(np.abs(2*ke/pe))

######## Energy increment functions ########
#@profile
def EnergyIncrement(
    i, m, M,
    x, v, u, h,
    v_com,
    tree=None,
    particles_not_in_tree=None):

    phi = 0.
    xtarget = np.array([x[i],])
    if len(particles_not_in_tree):
        ## have to get potential from particles not in the tree by brute force
        arr = particles_not_in_tree
        xa = arr[:,:3] #np.take(x, particles_not_in_tree_a,axis=0)
        ma = arr[:,3] #np.take(m, particles_not_in_tree_a)
        ha = arr[:,4] #np.take(h, particles_not_in_tree_a)
        phi += pytreegrav.PotentialTarget(xtarget, xa, ma, h_source=ha, G=4.3007e3,parallel=pytreegrav_parallel)[0]
    if tree:
        phi += 4.3007e3 * pytreegrav.PotentialTarget(
            xtarget,
            None,None, ## source pos and mass
            tree=tree,
            theta=1,parallel=pytreegrav_parallel)[0]
            
    vSqr = np.sum((v[i]-v_com)*(v[i]-v_com))
    mu = m[i]*M/(m[i]+M)
    return 0.5*mu*vSqr + m[i]*u[i] + m[i]*phi

def KE_Increment(
    i,
    m, v, u,
    v_com,
    mtot):

    vSqr = np.sum((v[i]-v_com)*(v[i]-v_com))
    mu = m[i]*mtot/(m[i]+mtot)
    return 0.5*mu*vSqr + m[i]*u[i]

def PE_Increment(
    i,
    c, m, x, v, u,
    v_com):

    phi = -4.3007e3 * np.sum(m[c]/cdist([x[i],],x[c]))
    return m[i]*phi

######## Grouping functions ########
@profile
def ParticleGroups(x, m, rho, h, u, v, nmin, ntree, alpha_crit, cluster_ngb=32, rmax=1e100):
    ngbdist, ngb = cKDTree(x).query(x,min(cluster_ngb, len(x)), distance_upper_bound=min(rmax, h.max()))

    max_group_size = 0
    groups = {}
    particles_since_last_tree = {}
    group_tree = {}
    group_alpha_history = {}
    group_energy = {}
    group_KE = {}
    group_PE = {}
    COM = {}
    v_COM = {}
    masses = {}
    positions = {}
    softenings = {}
    bound_groups = {}
    num_crossings = {}
    bound_subgroups = {}
    alpha_vir = {}
    assigned_group = -np.ones(len(x),dtype=np.int32)
    
    assigned_bound_group = -np.ones(len(x),dtype=np.int32)
    largest_assigned_group = -np.ones(len(x),dtype=np.int32)
    
    for i in range(len(x)):
        avir = 1e100
        ## do it one particle at a time, in decreasing order of density
#        if not i%10000:
#            print("Processed %d of %g particles; ~%3.2g%% done."%(i, len(x), 100*(float(i)/len(x))**2))
        if np.any(ngb[i] > len(x) -1):
            groups[i] = [i,]
            group_tree[i] = None
            assigned_group[i] = i
            group_energy[i] = m[i]*u[i]
            group_KE[i] = m[i]*u[i]
            group_PE[i] = 0
            v_COM[i] = v[i]
            COM[i] = x[i]
            masses[i] = m[i]
            particles_since_last_tree[i] = np.array([[x[i,0], x[i,1],x[i,2],m[i],h[i]]])
            alpha_vir[i] = 1e100
            num_crossings[i] = 0
            continue 
        ngbi = ngb[i][1:]

        denser = rho[ngbi] > rho[i]
        if denser.sum():
            ngb_denser, ngbdist_denser = ngbi[denser], ngbdist[i][1:][denser]
            ngb_denser = ngb_denser[ngbdist_denser.argsort()]
            ndenser = len(ngb_denser)
        else:
            ndenser = 0

        add_to_existing_group = False
        if ndenser == 0: # if this is the densest particle in the kernel, let's create our own group
            groups[i] = [i,]
            group_tree[i] = None
            assigned_group[i] = i
            group_energy[i] = m[i]*u[i]# - 2.8*m[i]**2/h[i] / 2 # kinetic + potential energy
            group_KE[i] = m[i]*u[i]
            v_COM[i] = v[i]
            COM[i] = x[i]
            masses[i] = m[i]
            alpha_vir[i] = 1e100
            particles_since_last_tree[i] = np.array([[x[i,0], x[i,1],x[i,2],m[i],h[i]]])
            num_crossings[i] = 0
        # if there is only one denser particle, or both of the nearest two denser ones belong to the same group, we belong to that group too
        elif ndenser == 1 or assigned_group[ngb_denser[0]] == assigned_group[ngb_denser[1]]: 
            assigned_group[i] = assigned_group[ngb_denser[0]]
            groups[assigned_group[i]].append(i)
            add_to_existing_group = True
        # o fuck we're at a saddle point, let's consider both respective groups
        else: 
            a, b = ngb_denser[:2]
            group_index_a, group_index_b = assigned_group[a], assigned_group[b]
            # make sure group a is the bigger one, switching labels if needed
            if masses[group_index_a] < masses[group_index_b]: group_index_a, group_index_b = group_index_b, group_index_a 

            # if both dense boyes belong to the same group, that's the group for us too
            if group_index_a == group_index_b:  
                assigned_group[i] = group_index_a
            #OK, we're at a saddle point, so we need to merge those groups
            else:
                group_a, group_b = groups[group_index_a], groups[group_index_b] 
                ma, mb = masses[group_index_a], masses[group_index_b]
                xa, xb = COM[group_index_a], COM[group_index_b] 
                va, vb = v_COM[group_index_a], v_COM[group_index_b]
                group_ab = group_a + group_b
                groups[group_index_a] = group_ab

                group_energy[group_index_a] += group_energy[group_index_b]
                group_KE[group_index_a] += group_KE[group_index_b]
                group_KE[group_index_a] += 0.5*ma*mb/(ma+mb) * np.sum((va-vb)**2)
                group_energy[group_index_a] += 0.5*ma*mb/(ma+mb) * np.sum((va-vb)**2) # energy due to relative motion: 1/2 * mu * dv^2
                if num_crossings[group_index_a] + num_crossings[group_index_b] <= max_num_crossings: # only need to keep track of energy if we are <= max_num_crossings
                # mutual interaction energy; we've already counted their individual binding energies
                    group_energy[group_index_a] += InteractionEnergy( 
                        x,m,h,
                        group_a,group_tree[group_index_a],
                        particles_since_last_tree[group_index_a],
                        group_b,group_tree[group_index_b],
                        particles_since_last_tree[group_index_b]) 

                    # we've got a big group, so we should probably do stuff with the tree
                    if len(group_a) > ntree: 
                        # if the smaller of the two is also large, let's build a whole new tree, and a whole new adventure
                        if len(group_b) > 512: 
                            group_tree[group_index_a] = pytreegrav.ConstructTree(
                                np.take(x,group_ab,axis=0),
                                np.take(m,group_ab),
                                np.take(h,group_ab))
                            particles_since_last_tree[group_index_a] = np.zeros((0,5))
                        # otherwise we want to keep the old tree from group a, and just add group b to the list of particles_since_last_tree
                        else:  
                            particles_since_last_tree[group_index_a] = np.concatenate([particles_since_last_tree[group_index_a], particles_since_last_tree[group_index_b]])
                    else:
                        particles_since_last_tree[group_index_a] = np.concatenate([particles_since_last_tree[group_index_a], particles_since_last_tree[group_index_b]])
#                        particles_since_last_tree[group_index_a][:] = [np.array([x[k,0], x[k,1],x[k,2],m[k],h[k]]) for k in group_ab]

                    if len(particles_since_last_tree[group_index_a]) > ntree:
                        group_tree[group_index_a] = pytreegrav.ConstructTree(
                            np.take(x,group_ab,axis=0),
                            np.take(m,group_ab),
                            np.take(h,group_ab))
 
                        particles_since_last_tree[group_index_a] = np.empty((0,5))
                else: # if we are > max_num_crossings
                    group_energy[group_index_a] = group_KE[group_index_a]                    
                                        
                COM[group_index_a] = (ma*xa + mb*xb)/(ma+mb)
                v_COM[group_index_a] = (ma*va + mb*vb)/(ma+mb)
                masses[group_index_a] = ma + mb
                num_crossings[group_index_a] = num_crossings[group_index_a] + num_crossings[group_index_b]
                groups.pop(group_index_b,None)
                assigned_group[i] = group_index_a
                assigned_group[assigned_group==group_index_b] = group_index_a

                avir_old = alpha_vir[group_index_a]
                # if this new group is bound, we can delete the old bound group
                if num_crossings[group_index_a] <= max_num_crossings: avir = abs(2*group_KE[group_index_a]/np.abs(group_energy[group_index_a] - group_KE[group_index_a]))
                else: avir=1e100
                alpha_vir[group_index_a] = avir

                if avir_old > alpha_crit and avir < alpha_crit and len(group_ab) > cluster_ngb:
                    num_crossings[group_index_a] += 1
                if avir < alpha_crit and num_crossings[g] <= max_num_crossings:
                    largest_assigned_group[group_ab] = len(group_ab)
                    assigned_bound_group[group_ab] = group_index_a
                for d in groups, particles_since_last_tree, group_tree, group_energy, group_KE, COM, v_COM, masses, alpha_vir: # delete the data from the absorbed group
                    d.pop(group_index_b, None)
                add_to_existing_group = True
#                if((avir_old-alpha_crit)*(avir-alpha_crit) < 0): print("crossing %g %g"%(avir_old, avir))
                
            groups[group_index_a].append(i)
            max_group_size = max(max_group_size, len(groups[group_index_a]))

            
        # assuming we've added a particle to an existing group, we have to update stuff
        if add_to_existing_group: 
            g = assigned_group[i]
            mgroup = masses[g]
            if num_crossings[g] <= max_num_crossings:
                group_KE[g] += KE_Increment(i, m, v, u, v_COM[g], mgroup)
                group_energy[g] += EnergyIncrement(i, m, mgroup, x, v, u, h, v_COM[g], group_tree[g], particles_since_last_tree[g]) 
            avir_old = alpha_vir[g]            
            if num_crossings[g] <= max_num_crossings: avir = abs(2*group_KE[g]/np.abs(group_energy[g] - group_KE[g]))
            else: avir = 1e100
            alpha_vir[g] = avir
            if avir_old > alpha_crit and avir < alpha_crit and len(groups[g]) > cluster_ngb:
                num_crossings[g] += 1
            if avir < alpha_crit and num_crossings[g] <= max_num_crossings:
                largest_assigned_group[i] = len(groups[g])
                assigned_bound_group[groups[g]] = g
                assigned_bound_group[i] = g
            v_COM[g] = (m[i]*v[i] + mgroup*v_COM[g])/(m[i]+mgroup)
            masses[g] += m[i]            
            particles_since_last_tree[g] = np.vstack((particles_since_last_tree[g],np.array([x[i,0], x[i,1],x[i,2],m[i],h[i]])))
            if len(particles_since_last_tree[g]) > ntree and num_crossings[g] <= max_num_crossings:
                group_tree[g] = pytreegrav.ConstructTree(np.take(x,groups[g],axis=0), np.take(m,groups[g]), np.take(h,groups[g]))
                particles_since_last_tree[g] = np.zeros((0,5))
            max_group_size = max(max_group_size, len(groups[g]))
#            if((avir_old-alpha_crit)*(avir-alpha_crit) < 0): print("crossing %g %g"%(avir_old, avir))

    # Now assign particles to their respective bound groups
    for i in range(len(assigned_bound_group)):
        a = assigned_bound_group[i]
        if a < 0:
            continue

        if a in bound_groups.keys(): bound_groups[a].append(i)
        else: bound_groups[a] = [i,]
#    print([VirialParameter(bound_groups[k], x, m, h, v, u) for k in bound_groups.keys()])

    return groups, bound_groups, assigned_group

    
def ComputeClouds(filename, options):
    print(filename)
    
    cluster_ngb = int(float(options["--cluster_ngb"]) + 0.5)
    G = float(options["--G"])
    alpha_crit = float(options["--alpha_crit"])
    boxsize = options["--boxsize"]
    nmin = float(options["--nmin"])
    recompute_potential = options["--recompute_potential"]
    ignore_B = options["--ignore_B"]
    ptype = "PartType0"
    
    outdir = filename.split("snapshot")[0] + "cores"
    if not path.isdir(outdir): mkdir(outdir)
    n = filename.split(".hdf5")[0].split("_")[-1]
    fname = outdir + "/Cores_%s_n%g_a%g_B%d.hdf5"%(n,nmin,alpha_crit, not ignore_B)
    if path.isfile(fname): return

    if boxsize != "None":
        boxsize = float(boxsize)
    else:
        boxsize = None


    if path.isfile(filename):
        F = h5py.File(filename,'r')
    else:
        print(("Could not find "+filename))
        return
    if not ptype in list(F.keys()):
        print("Particles of desired type not found!")
        
    time = F["Header"].attrs["Time"]
    m = np.array(F[ptype]["Masses"])
    criteria = np.ones(len(m),dtype=np.bool)

    if len(m) < 32:
        print("Not enough particles for meaningful cluster analysis!")
        return

    x = np.array(F[ptype]["Coordinates"])
    ids = np.array(F[ptype]["ParticleIDs"])
    u = np.array(F[ptype]["InternalEnergy"])
    rho = np.array(F[ptype]["Density"])
    B = np.array(F[ptype]["MagneticField"])
 
    criteria *= (rho*29.9 > nmin) # only look at dense gas (>nmin cm^-3) 145.7

    print(("%g particles denser than %g cm^-3" %(np.sum(rho*29.9>nmin), nmin)))
    if criteria.sum() < cluster_ngb: return

    m = m[criteria]
    x = x[criteria]
    u = u[criteria]
    v = np.array(F[ptype]["Velocities"])[criteria]
    rho = rho[criteria]
    ids = ids[criteria]
    if not ignore_B:
        vA = np.sqrt((B[criteria]**2).sum(1) / (rho)) * 3.429e12
        u += 0.5*vA*vA  # add magnetic energy to internal energy
    h = np.array(F[ptype]["SmoothingLength"])[criteria]
    x, m, rho, h, u, v = np.float64(x), np.float64(m), np.float64(rho), np.float64(h), np.float64(u), np.float64(v)

    phi = -rho
    order = phi.argsort()
    phi[:] = phi[order]
    x[:], m[:], v[:], h[:], u[:], rho[:], ids[:] = x[order], m[order], v[order], h[order], u[order], rho[order], ids[order]
    ngbdist, ngb = cKDTree(x).query(x,cluster_ngb)
    groups, bound_groups, assigned_group = ParticleGroups(x, m, rho, h, u, v, nmin, ntree, alpha_crit, cluster_ngb)#: ParticleGroups(x, m, rho, h, u, v, ngb, ngbdist,des_ngb=cluster_ngb)
    groupmass = np.array([m[c].sum() for c in list(bound_groups.values()) if len(c)>10])
    groupid = np.array([c for c in list(bound_groups.keys()) if len(bound_groups[c])>10])
    groupid = groupid[groupmass.argsort()[::-1]]
    bound_groups = OrderedDict(list(zip(groupid, [bound_groups[i] for i in groupid])))
#    exit()

    # Now we analyze the clouds and dump their properties

    bound_data = OrderedDict()
    bound_data["Mass"] = []
    bound_data["CenterOfMass"] = []
    bound_data["DensityPeak"] = []
    bound_data["PrincipalAxes"] = []
    bound_data["Reff"] = []
    bound_data["HalfMassRadius"] = []
    bound_data["NumParticles"] = []
#    bound_data["SigmaEff"] = []
    bound_data["Vdisp"] = []
    bound_data["Alpha"] = []

#    bound_data["SFE"] = []
#    bound_data["epsff"] = []
    

    i = 0
    fids = np.array(F["PartType0"]["ParticleIDs"])
    Fout = h5py.File(fname, 'w')
    # have to prefetch 
    for k,c in list(bound_groups.items()):
        if len(c) < 32: continue
        bound_data["Mass"].append(m[c].sum())
        bound_data["NumParticles"].append(len(c))
        bound_data["CenterOfMass"].append(np.average(x[c], weights=m[c], axis=0))
        bound_data["DensityPeak"].append(x[c][rho[c].argmax()])
        dx = x[c] - bound_data["CenterOfMass"][-1]
        eig = np.linalg.eig(np.cov(dx.T))[0]
        bound_data["PrincipalAxes"].append(np.sqrt(eig))
        bound_data["Reff"].append(np.prod(np.sqrt(eig))**(1./3))
        r = np.sum(dx**2, axis=1)**0.5
        bound_data["HalfMassRadius"].append(np.median(r))
        dv = v[c] - np.average(v[c], weights=m[c], axis=0)
        bound_data["Vdisp"].append(np.mean((dv*dv).sum(1))**0.5)
        bound_data["Alpha"].append(VirialParameter(c, x, m, h, v, u))
        cluster_id = "Cloud"+ ("%d"%i).zfill(int(np.log10(len(bound_groups))+1))
        i += 1
        N = len(c)




        if Fout is not None:
            idx = np.in1d(fids, ids[c])
            Fout.create_group(cluster_id)
            #for k in list(F["PartType0"].keys()):
            Fout[cluster_id].create_dataset("PartType0/ParticleIDs", data = fids[idx])  # data = np.array(F["PartType0"][k])[idx])
# below is for associating sink particles with subclouds
    #     if "epsff" in bound_data.keys():
    #         if "PartType5" in list(F.keys()):
    #             x_stars = np.array(F["PartType5"]["Coordinates"])
    #             mstar = np.array(F["PartType5"]["Masses"])
    #             tf = np.array(F["PartType5"]["StellarFormationTime"])
    #             yso = (time - tf)*979 < 0.5
    #             within_2sigma = np.sum((x_stars - np.median(x[c], axis=0))**2, axis=1)**0.5 < 2*bound_data["HalfMassRadius"][-1] * yso
    #             bound_data["SFE"].append(mstar[within_2sigma].sum()/(mstar[within_2sigma].sum() + bound_data["Mass"][-1]))
    #             rho0 = 0.5*bound_data["Mass"][-1]/ (4* np.pi * bound_data["HalfMassRadius"][-1]**3 / 3)
    #             tff0 = (3*np.pi/(32*4300.77*rho0))**0.5 * 979

    # #            print(mstar[within_2sigma].sum() / 0.5 /(bound_data["Mass"][-1]/tff0))
    #             bound_data["epsff"].append(mstar[within_2sigma].sum() / 0.5 /(bound_data["Mass"][-1]/tff0))
    #         else:
    #             bound_data["SFE"].append(0)
    #             bound_data["epsff"].append(0)
            #            if np.any(within_2sigma):            
#                for k in list(F["PartType5"].keys()):
#                    Fout[cluster_id].create_dataset("PartType5/"+k, data = np.array(F["PartType5"][k])[within_2sigma])
        
#        i += 1
        

    print("Done grouping bound clusters!")

        
    if Fout is not None: Fout.close()
    F.close()
    
    #now save the ascii data files
    SaveArrayDict(outdir+"/Cores_%s_n%g_a%g_B%d.dat"%(n,nmin,alpha_crit, not ignore_B), bound_data)
#    SaveArrayDict(filename.split("snapshot")[0] + "unbound_%s.dat"%n, unbound_data)

from multiprocessing import Pool
    
from natsort import natsorted
def main():
    options = docopt(__doc__)
#    for f in options["<files>"]:
#        print(f)
#        ComputeClouds(f, options)
    set_num_threads(int(options["--ngrav"]))
    if options["--np"] == 1:
        [ComputeClouds(f, options) for f in options["<files>"]]
    else:
        Parallel(n_jobs=int(options["--np"]))(delayed(ComputeClouds)(f, options) for f in natsorted(options["<files>"]))

if __name__ == "__main__": main()
