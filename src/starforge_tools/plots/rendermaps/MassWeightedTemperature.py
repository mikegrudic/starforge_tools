"""Mass-weighted temperature map"""

from meshoid import Meshoid
from .rendermap import RenderMap


class MassWeightedTemperature(RenderMap):
    plotlabel = r"$T \,\left(\rm K\right)$"
    required_datafields = (
        "PartType0/Coordinates",
        "PartType0/Masses",
        "PartType0/SmoothingLength",
        "PartType0/Temperature",
    )
    colormap = "plasma"
    cmap_default_limits = (5, 5e5)

    @staticmethod
    def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
        sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
        Tsigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Temperature"], **mapargs)
        return Tsigma / sigma
