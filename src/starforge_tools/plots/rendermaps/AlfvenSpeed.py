"""Gas surface density map"""

from meshoid import Meshoid
import matplotlib.colors as colors
import numpy as np

plotlabel = r"$\Sigma_{\rm gas}\,\left(M_\odot\,\rm pc^{-2}\right)$"  # label that will appear on the colorbar
required_datafields = {
    "PartType0/Coordinates",
    "PartType0/Masses",
    "PartType0/SmoothingLength",
}  # additional datafields beyond just the basic coordinates and smoothing length
colormap = "viridis"


# function called render that makes the actual map from particle data, a meshoid constructed from the data, and arguments to meshoid rendering functions
def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
    """Simple gas surface density map"""
    return meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)


# function that returns the limits for the colormap
def cmap_default_limits(map):
    """For surface density map: choose limits that leave 99% of the mass unsaturated in the colormap"""
    flatmap = np.sort(map.flatten())
    return np.interp([0.01, 0.99], flatmap.cumsum() / flatmap.sum(), flatmap)
