"""Gas velocity dispersion map"""

from meshoid import Meshoid
import numpy as np
from .rendermap import RenderMap


class VelocityDispersion(RenderMap):
    plotlabel = r"$\sigma_{\rm 1D}\,\left(\rm km\,s^{-1}\right)$"
    required_datafields = (
        "PartType0/Coordinates",
        "PartType0/Masses",
        "PartType0/SmoothingLength",
        "PartType0/Velocities",
    )
    colormap = "magma"
    cmap_default_limits = (0.3, 30)

    @staticmethod
    def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
        sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
        vSqr_map = (
            meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Velocities"][:, 2] ** 2, **mapargs)
            / sigma
        )
        vmap = (
            meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Velocities"][:, 2], **mapargs) / sigma
        )
        return np.sqrt(np.clip(vSqr_map - vmap**2, 0, 1e100)) / 1e3
