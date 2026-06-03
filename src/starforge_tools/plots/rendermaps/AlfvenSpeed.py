"""Mass-weighted RMS Alfven speed map"""

from meshoid import Meshoid
import numpy as np
from .rendermap import RenderMap


class AlfvenSpeed(RenderMap):
    plotlabel = r"$v_{\rm A}\,\left(\rm km\,s^{-1}\right)$"
    required_datafields = (
        "PartType0/Coordinates",
        "PartType0/Masses",
        "PartType0/SmoothingLength",
        "PartType0/MagneticField",
    )
    colormap = "RdYlBu"
    cmap_default_limits = (0.3, 30)

    @staticmethod
    def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
        pos = pdata["PartType0/Coordinates"]
        boxsize = pdata["Header"]["BoxSize"]
        cut = np.abs(pos[:, 2] - 0.5 * boxsize) < 100 * 0.15 * boxsize
        Bsqr = np.sum(pdata["PartType0/MagneticField"] ** 2, axis=1)
        tesla2_to_egydens = 5.88e18  # 1/(2 mu0) in our units
        sigma_Emag = meshoid.Projection((Bsqr * tesla2_to_egydens) * cut, **mapargs)
        sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"] * cut, **mapargs)
        return np.sqrt(2 * sigma_Emag / sigma)
