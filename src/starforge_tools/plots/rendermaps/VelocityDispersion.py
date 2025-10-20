"""Gas velocity dispersion map density map"""

from meshoid import Meshoid
import numpy as np

plotlabel = r"$\sigma_{\rm 1D}\,\left(\rm km\,s^{-1}\right)$"  # label that will appear on the colorbar
required_datafields = {
    "PartType0/Coordinates",
    "PartType0/Masses",
    "PartType0/SmoothingLength",
    "PartType0/Velocities",
}  # additional datafields beyond just the basic coordinates and smoothing length
colormap = "magma"


# function called render that makes the actual map from particle data,
# a meshoid constructed from the data, and arguments to meshoid rendering functions
def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
    sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
    vSqr_map = (
        meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Velocities"][:, 2] ** 2, **mapargs) / sigma
    )
    vmap = meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Velocities"][:, 2], **mapargs) / sigma
    return np.sqrt(np.clip(vSqr_map - vmap**2, 0, 1e100)) / 1e3


# function that returns the default limits for the colormap if none are provided
def cmap_default_limits(map):
    return [0.3, 30]
