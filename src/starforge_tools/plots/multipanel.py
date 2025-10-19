"""Implements a matplotlib plot consisting of multiple panels, with a timelapse from left to right and
stacking the different maps atop one another.
"""

from glob import glob
from os.path import isfile
from natsort import natsorted
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import h5py
from astropy import units as u
from scipy.spatial import KDTree
import numpy as np
from .map_renderer import MapRenderer
import rendermaps

DEFAULT_MAPS = ("SurfaceDensity", "VelocityDispersion", "MassWeightedTemperature", "AlfvenSpeed")


def get_pdata_for_maps(snapshot_path: str, maps=DEFAULT_MAPS) -> dict:
    """Does the I/O to get the data required for the specified maps"""
    required_data = set.union(*[getattr(rendermaps, s).required_datafields for s in maps])
    snapdata = {}
    with h5py.File(snapshot_path, "r") as F:
        snapdata["Header"] = dict(F["Header"].attrs)
        for i in range(6):
            s = f"PartType{i}"
            if s not in F.keys():
                continue
            for k in F[s].keys():
                s2 = s + "/" + k
                if s2 in required_data or i == 5:
                    snapdata[s2] = F[s2][:]

    return snapdata


def get_snapshot_units(F):
    unit_length = F["Header"].attrs["UnitLength_In_CGS"] * u.cm
    unit_speed = F["Header"].attrs["UnitVelocity_In_CGS"] * u.cm / u.s
    unit_mass = F["Header"].attrs["UnitMass_In_CGS"] * u.g
    unit_magnetic_field = 1e4 * u.gauss
    return {"Length": unit_length, "Speed": unit_speed, "Mass": unit_mass, "MagneticField": unit_magnetic_field}


def get_snapshot_timeline(output_dir, verbose=False):
    times = []
    snappaths = []
    if verbose:
        print("Getting snapnum timeline...")
    snappaths = natsorted(glob(output_dir + "/snapshot*.hdf5"))
    if not snappaths:
        raise FileNotFoundError(f"No snapshots found in {output_dir}")

    timelinepath = output_dir + "/.timeline"
    if isfile(timelinepath):  # check if we have a cached timeline file
        times = np.load(timelinepath)

    with h5py.File(snappaths[0], "r") as F:
        units = get_snapshot_units(F)

    if len(times) < len(snappaths):
        for f in snappaths:
            with h5py.File(f, "r") as F:
                times.append(F["Header"].attrs["Time"])
    if verbose:
        print("Done!")
    np.save(timelinepath, np.array(times))
    return np.array(snappaths), np.array(times) * (units["Length"] / units["Speed"]).to(u.Myr)


def multipanel_timelapse_map(output_dir=".", maps=DEFAULT_MAPS, times=4, res=2048, box_frac=0.24):
    snappaths, snaptimes = get_snapshot_timeline(output_dir)

    if isinstance(times, int):  # if we specified an integer number of times, assume evenly-spaced
        snaps = snappaths[:: len(snappaths) // (times - 1)]
        times = snaptimes[:: len(snaptimes) // (times - 1)]
    elif len(times):  # eventually implement nearest-neighbor snapshots of specified times
        ngb_time, ngb_idx = KDTree(np.c_[snaptimes]).query(np.c_[times])
        times = ngb_time
        snaps = snappaths[ngb_idx]
    else:
        raise NotImplementedError("Format not recognized for supplied times for multipanel map.")

    num_maps, num_times = len(maps), len(times)
    fig, ax = plt.subplots(num_maps, num_times, figsize=(8, 8))
    for i in range(len(times)):
        pdata = get_pdata_for_maps(snaps[i], maps)
        boxsize = pdata["Header"]["BoxSize"]
        length = boxsize * box_frac
        mapargs = {"size": length, "res": res}
        X, Y = 2 * [np.linspace(-0.5 * length, 0.5 * length, res)]
        if "PartType5/Masses" in pdata:
            SFE = pdata["PartType5/Masses"].sum() / (0.8 * pdata["PartType0/Masses"].sum())
        else:
            SFE = 0
        ax[0, i].set(title=f"{round(times[i],1)}Myr, SFE={round(SFE*100,0)}\%")
        renderer = MapRenderer(pdata, mapargs)
        for j, mapname in enumerate(maps):
            axes = ax[j, i]
            render = renderer.get_render(mapname)
            limits = renderer.limits[mapname]
            cmap = renderer.cmap[mapname]
            axes.pcolormesh(X, Y, render, norm=colors.LogNorm(*limits), cmap=cmap)

    fig.subplots_adjust(hspace=-0.0, wspace=0)
    plt.savefig("multipanel.png")


# rmax = 12
# res = 2048

# cmap = 'magma'

# fig, axes = plt.subplots(4,4,figsize=(8,8),sharex=True,sharey=True)

# axes[0,0].set(xlim=[-rmax,rmax], ylim=[-rmax,rmax])
# axes[0,0].set_aspect('equal')
# fig.subplots_adjust(hspace=-0.0,wspace=0)

# for j in range(4):
#     mass, B, v, pos, hsml, xs, mstar, T, vA = masses[j], Bs[j], vs[j], positions[j], hsmls[j], xss[j], mstars[j], Ts[j], vAs[j]
#     print(mass.shape, B.shape)
#     time = 979 * h5py.File(snaps[j],'r')["Header"].attrs["Time"]
#     xs, mstar = xs[mstar.argsort()][::-1], np.sort(mstar)[::-1]
#     SFE = mstar.sum() / 2e4
#     starcolors = np.array([np.interp(np.log10(mstar),[-1,0,1],star_colors[:,i]) for i in range(3)]).T
#     X = Y = np.linspace(-rmax, rmax, res+1)
#     X, Y = np.meshgrid(X, Y)

#     filename = "maps_%d_res%d.pickle"%(j,res)
#     if not isfile(filename):
#         hsml = np.clip(hsml,(2*rmax/res),1e100)
#         M = Meshoid(pos,mass,hsml,n_jobs=-1)
#         zmax = 20
#         sigma_gas = M.SurfaceDensity(center=np.array([0,0,0]),size=2*rmax,res=res).T
#         sigma_v = M.SurfaceDensity(mass*v[:,2]**2*(np.abs(pos[:,2])<zmax),center=np.array([0,0,0]),size=2*rmax,res=res).T/sigma_gas
#         v_avg = M.SurfaceDensity(mass*v[:,2]*(np.abs(pos[:,2])<zmax),center=np.array([0,0,0]),size=2*rmax,res=res).T/sigma_gas
#         v_avg = M.SurfaceDensity(mass*v[:,2]*(np.abs(pos[:,2])<zmax),center=np.array([0,0,0]),size=2*rmax,res=res).T/sigma_gas
#         vx = M.SurfaceDensity(mass*v[:,0]*(np.abs(pos[:,2])<zmax),center=np.array([0,0,0]),size=2*rmax,res=res).T/sigma_gas
#         vy = M.SurfaceDensity(mass*v[:,1]*(np.abs(pos[:,2])<zmax),center=np.array([0,0,0]),size=2*rmax,res=res).T/sigma_gas
#         sigma_v = (sigma_v - v_avg**2)**0.5
#         temp = M.SurfaceDensity(mass*T*(np.abs(pos[:,2])<zmax),center=np.array([0,0,0]),size=2*rmax,res=res).T/sigma_gas
#         Bmap = M.SurfaceDensity(mass*np.sum(vA**2,axis=1)*(np.abs(pos[:,2])<zmax),center=np.array([0,0,0]),size=2*rmax,res=res).T/sigma_gas
#         Q =  M.SurfaceDensity(mass*(B[:,0]**2-B[:,1]**2)/np.sum(B**2,axis=1),center=np.array([0,0,0]),size=2*rmax,res=res).T
#         U =  M.SurfaceDensity(mass*2*B[:,0]*B[:,1]/np.sum(B**2,axis=1),center=np.array([0,0,0]),size=2*rmax,res=res).T
#         with open(filename,'wb') as f: pickle.dump([sigma_gas,sigma_v,vx,vy,temp,Bmap,Q,U],f)
#     else:
#         with open(filename,'rb') as f: sigma_gas,sigma_v,vx,vy,temp,Bmap,Q,U = pickle.load(f)

#     for i in range(4):
#         cmap = {0: "viridis",1:"magma",2:"plasma",3:"RdYlBu"}[i]
#         limits = {0: [1,1e3], 1:[0.1,30], 2:[10,3e5], 3:[0.1,30]}[i]
#         ax = axes[i,j]
#         ax.set_aspect('equal')
#         if i==0:
#             p = ax.pcolormesh(X, Y, sigma_gas,norm=colors.LogNorm(vmin=limits[0],vmax=limits[1]),cmap=cmap,zorder=-1000);
#             ax.set_title("%3.2gMyr, SFE=%3.2g\%%"%(time,SFE*100))
#         if i==1:
#             p = ax.pcolormesh(X, Y, sigma_v,norm=colors.LogNorm(vmin=limits[0],vmax=limits[1]),cmap=cmap,zorder=-1000);
#             #tex = runlic(vX, vY,300)
#             #tex = 0.5 + (tex-tex.mean())/1.5
#             tex = get_lic_image(vx,vy)
#             tex = (tex-tex.min())/(tex.max()-tex.min())*0.4 + 0.5
#             color = plt.get_cmap(cmap)(np.log10(sigma_v/0.1)/2)
#             img = color*tex #LightSource().blend_hsv(color, tex)
#             ax.imshow(img[::-1],extent=(-rmax,rmax,-rmax,rmax))
#         if i==2: p = ax.pcolormesh(X, Y, temp, norm=colors.LogNorm(vmin=limits[0],vmax=limits[1]),cmap=cmap,zorder=-1000)
#         if i==3:
#             from matplotlib.colors import ListedColormap
#             import numpy as np
#             colombi1_cmap = ListedColormap(np.loadtxt("planck_parchment_rgb.txt")[::-1]/255.)
#             p = ax.pcolormesh(X, Y, Bmap ,cmap=colombi1_cmap ,norm=colors.LogNorm(vmin=limits[0],vmax=limits[1]),zorder=-1000)
#             phi = 0.5 * np.arctan2(U,Q)

#             vX, vY = np.cos(phi), np.sin(phi)
#             tex = get_lic_image(vX,vY)
#             #tex = (tex-tex.min())/(tex.max()-tex.min())#*0.5 + 0.5
#             color = plt.get_cmap(colombi1_cmap)(np.log10(Bmap/limits[0])/np.log10(limits[1]/limits[0]))
#             #img = tex  # color * (1+0.3*tex) #LightSource().blend_hsv(color, tex)
#             #ax.imshow(color[::-1,:,:-1]*tex[::-1][:,:,:-1],extent=(-rmax,rmax,-rmax,rmax),zorder=100)
#             #ax.imshow(color[::-1,:,:-1]+(1-tex[::-1][:,:,:-1]),extent=(-rmax,rmax,-rmax,rmax),zorder=100)
#             ax.imshow(color[::-1,:]*(tex[::-1]),extent=(-rmax,rmax,-rmax,rmax),zorder=0)
#             #X2 = Y2 = np.linspace(-rmax, rmax, res)
#             #X2, Y2 = np.meshgrid(X2, Y2)
#             #p2 = ax.streamplot(X2,Y2, vX,vY, color='black',linewidth=0.1,arrowsize=0,density=3,zorder=-1)
#         if i==3: ax.set(xlabel=r'$x\rm  \,\left(pc\right)$')#,ylabel=r'$\rm Y \,\left(pc\right)$')
#         if j==0: ax.set(ylabel=r'$z\rm  \,\left(pc\right)$')
#         ax.scatter(xs[:,0],xs[:,1],s=4*(mstar/1)**(0.5),edgecolor='black',facecolor=starcolors,marker='*',lw=0.02,alpha=1)
#         if i==0 and j==0:
#             for m_dummy in 0.1,1,10,100:
#                 ax.scatter([-100,],[-100],s=4*(m_dummy)**(0.5), color=[np.interp(np.log10(m_dummy),[-1,0,1],star_colors[:,i]) for i in range(3)],label=r"$%gM_\odot$"%m_dummy,edgecolor='black',alpha=1,lw=0.02,marker='*')
#             ledge = ax.legend(loc=2,frameon=True,facecolor='black',labelspacing=0.1,fontsize=6,edgecolor='white')
#             ledge.get_frame().set_linewidth(0.5)
#             ledge.get_frame().set_alpha(0.5)
#             for text in ledge.get_texts(): text.set_color("white")
#         if i==0 and j==3:
#             addColorbar(ax,cmap,limits[0],limits[1],r"$\Sigma_{\rm gas}$ $(\rm M_\odot\,\rm pc^{-2})$",logflag=1,span_full_figure=False,tick_tuple=((1,10,100,1000),(1,10,100,1000)))
#         if i==1 and j==3:
#             addColorbar(ax,cmap,limits[0],limits[1],r"$\sigma_{\rm 1D}$ $(\rm km\,s^{-1})$",logflag=1,span_full_figure=False,nticks=3)
#         if i==2 and j==3:
#             addColorbar(ax,cmap,limits[0],limits[1],r"$T$ $(\rm K)$",logflag=1,span_full_figure=False,nticks=5)
#         if i==3 and j==3:
#             addColorbar(ax,cmap,limits[0],limits[1],r"$v_{\rm A}$ $(\rm km\,s^{-1})$",logflag=1,span_full_figure=False,nticks=3)
#         #[i.set_linewidth(0.1) for i in ax.spines.values()]
# #         if i==2:
# #             print("doing relief")
# #             z = np.clip(np.log10(sigma_gas/1)/3,0,1)
# #             ls = LightSource(azdeg=315, altdeg=45)
# #             lightness = ls.hillshade(z, vert_exag=5)
# #             color = plt.get_cmap(cmap)(np.log10(sigma_v/0.1)/2)
# #             img = ls.blend_hsv(color[:,:,:3], lightness[:,:,None])
# #             ax.imshow(img[::-1],extent=(-rmax,rmax,-rmax,rmax))


# #[a.set_aspect('equal') for a in axes]

# fig.subplots_adjust(hspace=-0.0,wspace=0)
# plt.savefig("16panel_render.png",bbox_inches='tight',dpi=1000)#,dpi=300)
