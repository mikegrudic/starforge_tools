"""Gas surface density map"""

from meshoid import Meshoid
from .rendermap import RenderMap
import numpy as np


# function called render that makes the actual map from particle data,
# a meshoid constructed from the data, and arguments to meshoid rendering functions


# function that returns the limits for the colormap


class SurfaceDensity(RenderMap):
    plotlabel = r"$\Sigma_{\rm gas}\,\left(M_\odot\,\rm pc^{-2}\right)$"  # label that will appear on the colorbar
    required_datafields = (
        "PartType0/Coordinates",
        "PartType0/Masses",
        "PartType0/SmoothingLength",
    )  # required datafields to make this map
    colormap = "viridis"

    def render(self, pdata: dict, meshoid: Meshoid, mapargs: dict):
        """Simple gas surface density map"""
        return meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)

    @property
    def cmap_default_limits(self):
        """For surface density map: choose limits that leave 98% of the mass unsaturated in the colormap"""
        return [3, 3e3]
