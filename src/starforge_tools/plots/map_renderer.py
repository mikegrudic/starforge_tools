# maps container object has meshoid, pdata, mapargs, a dict of already rendered maps for quick access, and a method for resolving dependencies and rendering all desired maps
# map object has the implementation and dependencies on other maps

from meshoid import Meshoid
import numpy as np
from . import rendermap


class MapRenderer:
    def __init__(self, pdata: dict, mapargs: dict):
        self.pdata = pdata
        self.meshoid = Meshoid(
            pdata["PartType0/Coordinates"], pdata["PartType0/Masses"], pdata["PartType0/SmoothingLength"]
        )
        self.mapargs = mapargs
        self.rendered_maps = {}

    def render_map(self, map_name: str):
        self.check_if_map_implemented(map_name)
        map = getattr(rendermap, map_name)()
        self.rendered_maps[map_name] = map.render(self.pdata, self.meshoid, self.mapargs)

    def get_map(self, map_name: str) -> np.ndarray:
        self.check_if_map_implemented(map_name)
        if map_name not in self.rendered_maps:
            self.render_map(map_name)
        return self.rendered_maps[map_name]

    def check_if_map_implemented(self, map_name: str):
        """Checks if a map is implemented in multipanel_maps.py, and if not raises an error."""
        if map_name not in vars(rendermap):
            raise NotImplementedError(f"Requested map {map_name} has no renderer function in multipanel_maps.py")
