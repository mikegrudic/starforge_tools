"""Gas surface density map"""

from meshoid import Meshoid
import numpy as np

required_datafields = {
    "PartType0/Coordinates",
    "PartType0/Masses",
    "PartType0/SmoothingLength",
    "PartType0/MagneticField",
}  # additional datafields beyond just the basic coordinates and smoothing length
plotlabel = r"$v_{\rm A,RMS}\,\left(\rm km\,s^{-1}\right)$"
colormap = "RdYlBu"


# function called render that makes the actual map from particle data, a meshoid constructed from the data, and arguments to meshoid rendering functions
def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
    pos = pdata["PartType0/Coordinates"]
    boxsize = pdata["Header"]["BoxSize"]
    cut = np.abs(pos[:, 2] - 0.5 * boxsize) < 100 * 0.15 * boxsize
    Bsqr = np.sum(pdata["PartType0/MagneticField"] ** 2, axis=1)
    tesla2_to_egydens = 5.88e18  # 1/(2 mu0) in our units
    sigma_Emag = meshoid.Projection((Bsqr * tesla2_to_egydens) * cut, **mapargs)
    sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"] * cut, **mapargs)
    return np.sqrt(2 * sigma_Emag / sigma)


# function that returns the limits for the colormap
def cmap_default_limits(map):
    """For alfven speed: 0.3-30km/s"""
    return [0.3, 30]
