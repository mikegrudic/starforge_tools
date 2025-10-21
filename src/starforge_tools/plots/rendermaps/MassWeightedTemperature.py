"""Gas surface density map"""

from meshoid import Meshoid
import matplotlib.colors as colors
import numpy as np

plotlabel = r"$T \,\left(\rm K\right)$"  # label that will appear on the colorbar
required_datafields = {
    "PartType0/Coordinates",
    "PartType0/Masses",
    "PartType0/SmoothingLength",
    "PartType0/Temperature",
}  # additional datafields beyond just the basic coordinates and smoothing length
colormap = "plasma"


def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
    sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
    Tsigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Temperature"], **mapargs)
    return Tsigma / sigma


# function that returns the limits for the colormap
def cmap_default_limits(map):
    """For temp map: 5-5e5 K to span from coldest to hot gas"""
    return [5, 5e5]
