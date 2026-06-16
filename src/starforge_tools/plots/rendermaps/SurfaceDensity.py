"""Gas surface density map"""

from meshoid import Meshoid
from .rendermap import RenderMap


class SurfaceDensity(RenderMap):
    plotlabel = r"$\Sigma_{\rm gas}\,\left(M_\odot\,\rm pc^{-2}\right)$"
    required_datafields = (
        "PartType0/Coordinates",
        "PartType0/Masses",
        "PartType0/SmoothingLength",
    )
    colormap = "viridis"
    cmap_default_limits = (3, 3e3)

    @staticmethod
    def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
        return meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
