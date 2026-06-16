"""Map of the coordinate function Z (for orientation debugging purposes)"""

from meshoid import Meshoid
from .rendermap import RenderMap


class ZCoordinate(RenderMap):
    plotlabel = r"$z\,\left(\rm pc\right)$"
    required_datafields = (
        "PartType0/Coordinates",
        "PartType0/Masses",
        "PartType0/SmoothingLength",
    )
    colormap = "RdBu"
    cmap_default_limits = (None, None)
    log_cache = False  # coordinates can be negative

    @staticmethod
    def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
        """Mass-weighted projection of the z coordinate"""
        return meshoid.SurfaceDensity(
            pdata["PartType0/Masses"] * pdata["PartType0/Coordinates"][:, 1], **mapargs
        ) / meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
