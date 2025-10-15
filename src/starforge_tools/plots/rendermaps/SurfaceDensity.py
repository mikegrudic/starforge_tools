"""Gas surface density map"""

from meshoid import Meshoid

plotlabel = r"$\Sigma_{\rm gas}\,\left(M_\odot\,\rm pc^{-2}\right)$"  # label that will appear on the colorbar
required_datafields = "PartType0/Masses"  # additional datafields beyond just the basic coordinates and smoothing length


# function called render that makes the actual map from particle data, a meshoid constructed from the data, and arguments to meshoid rendering functions
def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
    return meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
