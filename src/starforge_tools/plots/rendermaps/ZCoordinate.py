"""Map of the coordinate function X (for orientation debugging purposes)"""

from meshoid import Meshoid

plotlabel = r"$z\,\left(\rm pc\right)$"  # label that will appear on the colorbar
required_datafields = {
    "PartType0/Coordinates",
    "PartType0/Masses",
    "PartType0/SmoothingLength",
}  # required datafields to make this map
colormap = "RdBu"


# function called render that makes the actual map from particle data,
# a meshoid constructed from the data, and arguments to meshoid rendering functions
def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
    """Slice of the coordinate function"""
    # return meshoid.Slice(pdata["PartType0/Coordinates"][:, 0], **mapargs, order=0)
    return meshoid.SurfaceDensity(
        pdata["PartType0/Masses"] * pdata["PartType0/Coordinates"][:, 1], **mapargs
    ) / meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)


# function that returns the limits for the colormap
def cmap_default_limits(map):
    return [None, None]  # [3, 3e3]


#    flatmap = np.sort(map.flatten())
#    return np.interp([0.01, 0.99], flatmap.cumsum() / flatmap.sum(), flatmap)
