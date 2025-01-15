"""Implementation of routine for computing the gas column density along stellar sightlines"""

import numpy as np
from scipy.spatial import KDTree


def star_gas_columns(xstar, xgas, mgas, hgas, sightline_dir=np.array([0, 0, -1.0])):
    """
    Returns the column density of gas of each star sightline along a specified sightline direction.

    Parameters
    ----------
    xstar: array_like
        shape (num_stars, 3) array of star coordinates
    xgas: array_like
        shape (num_gas, 3) array of gas coordinates
    mgas: array_like
        shape (num_gas,) array of gas masses; note that you can instead provide the
        cross section = opacity * mass if you wish to obtain optical depth instead of column density
    hgas: array_like
        shape (num_gas,) array of gas kernel lengths
    sightline_dir: array_like, optional
        Shape (3,) array specifying the sightline vector **from the observer to the stars** (default [0,0,-1])

    Returns
    -------
    columns: ndarray
        shape (num_stars,) array of column densities
    """
    if not np.allclose(
        sightline_dir, np.array([0, 0, -1.0])
    ):  # have to remap coordinates if sightline is not along +z direction
        if not isinstance(sightline_dir, np.ndarray):
            sightline_dir = np.array(sightline_dir, dtype=np.float64)
        sightline_dir /= (sightline_dir * sightline_dir).sum() ** 0.5

        # generate a random x axis, deproject z component, then cross to get new y axis
        sightline_x = np.random.normal(size=(3,))  # 3D random
        sightline_x -= np.inner(sightline_x, -sightline_dir) * sightline_dir  # deproject
        sightline_x /= (sightline_x * sightline_x).sum() ** 0.5  # normalize
        sightline_y = np.cross(-sightline_dir, sightline_x)  # cross product
        sightline_matrix = np.c_[sightline_x, sightline_y, -sightline_dir].T  # make unitary coordinate basis matrix
        xstar = xstar.copy() @ sightline_matrix  # convert to new basis - copy to avoid modifying in-place
        xgas = xgas.copy() @ sightline_matrix  # ditto

    # now make the search tree for stars
    star_tree = KDTree(xstar[:, :2])  # 2D tree
    gas_ngb_dist, gas_ngb = star_tree.query(xgas[:, :2], workers=-1)
    overlapping_star = (gas_ngb_dist < hgas) * (xstar[gas_ngb, 2] < xgas[:, 2])  # overlaps and in front

    # prune gas particles that do not overlap even 1 star
    xgas, mgas, hgas = xgas[overlapping_star], mgas[overlapping_star], hgas.copy()[overlapping_star]

    # now need to know every star that a gas particle overlaps
    ngb = star_tree.query_ball_point(xgas[:, :2], hgas, workers=-1)

    columns = np.zeros(xstar.shape[0])
    for i, n in enumerate(ngb):
        # loop over overlapping gas particles and deposit their columns - SLOW, OPTIMIZE ME FIRST
        columns[n] += mgas[i] / (np.pi * hgas[i] * hgas[i])

    return columns
