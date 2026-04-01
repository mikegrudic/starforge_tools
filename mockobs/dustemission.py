"""
Generates far-IR images of simulation snapshots that have dust temperature
information.

Usage: dustemission.py <files> ... [options]

Options:
   -h --help                   Show this screen.
   --size=<L>                  Image side length in pc (defaults to box size / 5)
   --center=<X,Y>              Center of the image (defaults to box center)
   --res=<N>                   Resolution of the image [default: 1024]
   --wavelengths=<l1,l2,etc>   Wavelengths in micron to image [default: 150, 250, 350, 500]
   --output_path=<path>        Output path for images (defaults to /dustemission 
                               directory next to the snapshot)
   --num_jobs=<N>              Number of snapshots to process in parallel [default: 1]
   --dir=<x,y,z>               Coordinate direction to orient the image along - x, y, or z. 
                               It also accepts vector values [default: z]
"""

from os import mkdir
from os.path import isdir
import pathlib
import numpy as np
from docopt import docopt
import h5py
from meshoid.radiation import (
    dust_emission_map,
    modified_blackbody_fit_image,
    modified_blackbody_fit_gaussnewton,
)
from meshoid.grid_deposition import GridSurfaceDensity
from joblib import Parallel, delayed

np.random.seed(42)

options = docopt(__doc__)
WAVELENGTHS = np.array([float(s) for s in options["--wavelengths"].split(",")])
RES = int(options["--res"])
if options["--size"]:
    SIZE = float(options["--size"])
else:
    SIZE = None
if options["--center"]:
    CENTER = np.array([float(s) for s in options["--center"].split(",")])
else:
    CENTER = None
if options["--output_path"]:
    if not isdir(options["--output_path"]):
        mkdir(options["--output_path"])
    OUTPATH = options["--output_path"]
else:
    OUTPATH = None
NUM_JOBS = int(options["--num_jobs"])
DIR = options["--dir"]

SOLAR_Z = 0.0142


def parse_dir_option(dir_option):
    """
    Parse the --dir option to return a 3-vector.
    Accepts 'x', 'y', 'z', or a comma-separated vector (e.g. '1,1,0').
    """
    if dir_option in ['x', 'y', 'z']:
        mapping = {'x': np.array([1, 0, 0]),
                   'y': np.array([0, 1, 0]),
                   'z': np.array([0, 0, 1])}
        return mapping[dir_option]
    else:
        # Assume comma-separated vector, e.g. "1,0,0"
        return np.array([float(x) for x in dir_option.split(',')])

def rotate_3d(coords, view_vector, center=None):
    """
    Rotate coords so view_vector aligns with z-axis, around a specified center.

    Parameters
    ----------
    coords : ndarray
        Nx3 array of coordinates.
    view_vector : array-like
        3-element vector to align with the z-axis.
    center : array-like or None
        3-element array, the center point to rotate around (default: origin).

    Returns
    -------
    rotated_coords : ndarray
        Rotated coordinates.
    """
    coords = np.asarray(coords)
    if center is not None:
        center = np.asarray(center)
        coords_shifted = coords - center
    else:
        coords_shifted = coords

    view_vector = view_vector / np.linalg.norm(view_vector)
    z_axis = np.array([0, 0, 1])
    v = np.cross(view_vector, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(view_vector, z_axis)
    I = np.eye(3)
    if s != 0:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    else:
        R = I
    rotated = np.dot(coords_shifted, R.T)
    if center is not None:
        rotated = rotated + center
    return rotated

def transform_coordinates(x, direction, center):
    """
    Transforms coordinates based on the specified direction or vector.

    Parameters
    ----------
    x : ndarray
        Nx3 array of coordinates.
    direction : str
        'x', 'y', 'z', or a comma-separated vector (e.g. '1,1,0').
    center : array-like
        3-element array, the center point to rotate around.

    Returns
    -------
    ndarray
        Transformed coordinates.
    """
    view_vector = parse_dir_option(direction)
    return rotate_3d(x, view_vector, center)

def make_dustemission_map_from_snapshot(path):
    """Makes a dust emission map from a STARFORGE snapshot"""
    with h5py.File(path, "r") as F:
        x = np.float32(F["PartType0/Coordinates"][:])
        m = np.float32(F["PartType0/Masses"][:])
        h = np.float32(F["PartType0/SmoothingLength"][:])
        Z = F["PartType0/Metallicity"][:]
        if len(Z.shape) == 2:
            Z = Z[:, 0] / SOLAR_Z
        else:
            Z = Z / SOLAR_Z

        if "PartType0/Dust_Temperature" in F.keys():
            Tdust = np.float32(F["PartType0/Dust_Temperature"][:])
        else:
            Tdust = np.repeat(20, len(x))
        Tdust_avg = np.average(Tdust, weights=m)

        boxsize = F["Header"].attrs["BoxSize"]
    if SIZE:
        size = SIZE
    else:
        size = 0.2 * boxsize
    if CENTER:
        center = CENTER
    else:
        center = 0.5 * np.array(3 * [boxsize])

    dx = size / (RES - 1)
    
    x = transform_coordinates(x, DIR, center)
    
    intensity = dust_emission_map(x, m * Z, h, Tdust, size, RES, WAVELENGTHS, center)

    # add noise
    # SNR = 60
    # noise_norm = intensity/SNR + 1e-18 * 10**(-np.log10(WAVELENGTHS[None,None,:]/500) * 2)
    # SNR_tot = noise_norm / intensity
    # N_eff = SNR_tot**-2
    # noise = np.random.poisson(N_eff)/N_eff * intensity - intensity
    h = h.clip(dx, 1e100)
    sigmagas = GridSurfaceDensity(m, x, h, center, size, RES)
    X = np.linspace(0.5 * (dx - size), 0.5 * (size - dx), RES) + center[0]
    Y = np.linspace(0.5 * (dx - size), 0.5 * (size - dx), RES) + center[1]
    X, Y = np.meshgrid(X, Y, indexing="ij")

    fname = path.split("/")[-1].replace(".hdf5", ".dustemission.hdf5")
    if OUTPATH:
        imgpath = OUTPATH + fname
    else:
        outdir = str(pathlib.Path(path).parent.resolve()) + "/dustemission/"
        if not isdir(outdir):
            mkdir(outdir)
        imgpath = outdir + fname
    with h5py.File(imgpath, "w") as F:
        F.create_dataset("Wavelengths_um", data=WAVELENGTHS)
        F.create_dataset("X_pc", data=X)
        F.create_dataset("Y_pc", data=Y)
        F.create_dataset("Intensity_cgs", data=intensity)
        #        F.create_dataset("Intensity_noise_cgs", data=noise)
        F.create_dataset("SurfaceDensity_Msun_pc2", data=sigmagas)

        # if len(WAVELENGTHS) >= 3:
        #     # fit each pixel to a modified blackbody as an observer would
        #     params = modified_blackbody_fit_image(intensity, 0.1*intensity, WAVELENGTHS)
        #     tau0, beta, Tdust_fit = (
        #         params[:, :, 0],
        #         params[:, :, 1],
        #         params[:, :, 2],
        #     )

        #     F.create_dataset("Fit_Tau500um", data=tau0)
        #     F.create_dataset("Fit_Beta", data=beta)
        #     F.create_dataset("Fit_Tdust", data=Tdust_fit)
        #     F.create_dataset("Tdust_massweighted", data=Tdust_avg)

        #     params_avg = modified_blackbody_fit_gaussnewton(
        #         intensity.mean((0, 1)), 0.1*intensity.mean((0,1)),WAVELENGTHS
        #     )

        #     tau0, beta, Tdust_fit = params_avg

        #     F.create_dataset("SEDFit_Tau500um", data=tau0)
        #     F.create_dataset("SEDFit_Beta", data=beta)
        #     F.create_dataset("SEDFit_Tdust", data=Tdust_fit)


def main():
    """Runs dust mapping and SED fitting on input snapshots"""
    Parallel(n_jobs=NUM_JOBS)(
        delayed(make_dustemission_map_from_snapshot)(f) for f in options["<files>"]
    )


if __name__ == "__main__":
    main()