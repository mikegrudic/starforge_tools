"""Map of the coordinate function X (for orientation debugging purposes)"""

from meshoid import Meshoid
from .rendermap import RenderMap


class XCoordinate(RenderMap):
    plotlabel = r"$x\,\left(\rm  pc\right)$"
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
        """Slice of the coordinate function"""
        return meshoid.Slice(pdata["PartType0/Coordinates"][:, 0], **mapargs, order=1)
