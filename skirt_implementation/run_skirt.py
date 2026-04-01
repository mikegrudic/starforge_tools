#!/usr/bin/env python3
"""
run_skirt.py

Create self-contained SKIRT input files from a simulation snapshot and run SKIRT.

Options:
   -h --help                   Show this screen.
   snapshot                    Path to HDF5 snapshot file.

   --output_path=DIR           Output directory for generated files and SKIRT outputs
                               [default: ./skirt_output]

   --stars=FILE                Output stars text filename [default: stars.txt]
   --gas=FILE                  Output gas text filename [default: gas.txt]
   --ski=FILE                  Output SKIRT parameter filename [default: run.ski]

   --center=X,Y,Z              Center of the image (defaults to box center)

   --box=MINX,MAXX,MINY,MAXY,MINZ,MAXZ
                               Simulation box bounds in pc
                               [default: 0,100,0,100,0,100]

   --fov=X[,Y]                 Instrument field of view in pc
                               [default: 25,25]

   --resolution=N[,M]          Instrument pixel resolution
                               [default: 1024,1024]

   --distance=D                Instrument distance in Mpc [default: 0.0005]

   --num_sites=N               Number of Voronoi sites [default: 500000]

   --num_packets=N             Number of photon packets [default: 5e7]

   --instrument=SPEC           Add an instrument. Can be repeated.
                               Allowed forms:
                                 x
                                 y
                                 z
                                 INC,AZ,ROLL
                               If omitted, defaults to:
                                 --instrument=z
                                 --instrument=x
                                 --instrument=y

   --instrument_wavelength_grid=SPEC
                               Shared wavelength grid for all instruments.
                               Allowed forms:
                                 WAVELENGTH
                                 MIN,MAX,NUM
                               [default: 0.1,2000,200]

   --mpi_processes=N           Number of MPI processes [default: 1]
   --threads=N                 Number of threads per MPI process [default: 1]

   --no_run_skirt              Only prepare files; do not run SKIRT.
"""

import argparse
import os
import subprocess
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np

code_time_to_Myrs = (u.pc / (u.m / u.s)).to(u.Myr)

ISRF_SED_DATA = """# Column 1: wavelength (micron)
# Column 2: specific luminosity (Lsun/micron)
1.000000e-03	0.000000e+00
1.075631e-03	0.000000e+00
1.156983e-03	0.000000e+00
1.244487e-03	0.000000e+00
1.338609e-03	0.000000e+00
1.439850e-03	0.000000e+00
1.548748e-03	0.000000e+00
1.665882e-03	0.000000e+00
1.791875e-03	0.000000e+00
1.927397e-03	0.000000e+00
2.073169e-03	0.000000e+00
2.229965e-03	0.000000e+00
2.398621e-03	0.000000e+00
2.580032e-03	0.000000e+00
2.775163e-03	0.000000e+00
2.985053e-03	0.000000e+00
3.210816e-03	1.092274e-310
3.453655e-03	1.542279e-292
3.714860e-03	1.135371e-275
3.995820e-03	5.363613e-260
4.298029e-03	1.972341e-245
4.623095e-03	6.755770e-232
4.972746e-03	2.546931e-219
5.348842e-03	1.234224e-207
5.753382e-03	8.880799e-197
6.188518e-03	1.085009e-186
6.656565e-03	2.549676e-177
7.160010e-03	1.294035e-168
7.701531e-03	1.579854e-160
8.284009e-03	5.128692e-153
8.910540e-03	4.859205e-146
9.584456e-03	1.465206e-139
1.030934e-02	1.523942e-133
1.108905e-02	5.892205e-128
1.192773e-02	9.079147e-123
1.282984e-02	5.947894e-118
1.380018e-02	1.759341e-113
1.484391e-02	2.484776e-109
1.596657e-02	1.765017e-105
1.717415e-02	6.617921e-102
1.847305e-02	1.369988e-98
1.987020e-02	1.632577e-95
2.137301e-02	1.164276e-92
2.298948e-02	5.151602e-90
2.472820e-02	1.462540e-87
2.659843e-02	2.748551e-85
2.861011e-02	3.519881e-83
3.077393e-02	3.155678e-81
3.310141e-02	2.030892e-79
3.560491e-02	9.603579e-78
3.829776e-02	3.409912e-76
4.119427e-02	9.276120e-75
4.430985e-02	1.969868e-73
4.766107e-02	3.322904e-72
5.126574e-02	4.525213e-71
5.514304e-02	5.050533e-70
5.931358e-02	4.684777e-69
6.379955e-02	3.658842e-68
6.862480e-02	2.435307e-67
7.381499e-02	1.397017e-66
7.939772e-02	6.979553e-66
8.540268e-02	3.066562e-65
9.186180e-02	1.195633e-64
9.880944e-02	4.171725e-64
1.062825e-01	1.312803e-63
1.143208e-01	3.753297e-63
1.229671e-01	9.815480e-63
1.322672e-01	2.363072e-62
1.422708e-01	5.269401e-62
1.530309e-01	1.094844e-61
1.646049e-01	2.132373e-61
1.770542e-01	3.917898e-61
1.904450e-01	6.838475e-61
2.048487e-01	1.142915e-60
2.203416e-01	1.845302e-60
2.370064e-01	2.905494e-60
2.549315e-01	4.501740e-60
2.742123e-01	6.912105e-60
2.949514e-01	1.055616e-59
3.172590e-01	1.603121e-59
3.412537e-01	2.412685e-59
3.670632e-01	3.580089e-59
3.948247e-01	5.210492e-59
4.246858e-01	7.405941e-59
4.568054e-01	1.024925e-58
4.913543e-01	1.378700e-58
5.285161e-01	1.801502e-58
5.684885e-01	2.286968e-58
6.114840e-01	2.822696e-58
6.577314e-01	3.391008e-58
7.074766e-01	3.970432e-58
7.609840e-01	4.537659e-58
8.185383e-01	5.069657e-58
8.804455e-01	5.545627e-58
9.470348e-01	5.948551e-58
1.018660e+00	6.266184e-58
1.095703e+00	6.491465e-58
1.178573e+00	6.622391e-58
1.267710e+00	6.661469e-58
1.363588e+00	6.614901e-58
1.466718e+00	6.491632e-58
1.577648e+00	6.302386e-58
1.696968e+00	6.058784e-58
1.825312e+00	5.772590e-58
1.963363e+00	5.455125e-58
2.111855e+00	5.116837e-58
2.271577e+00	4.767035e-58
2.443380e+00	4.413738e-58
2.628176e+00	4.063640e-58
2.826949e+00	3.722144e-58
3.040755e+00	3.393459e-58
3.270731e+00	3.080716e-58
3.518101e+00	2.786113e-58
3.784180e+00	2.511050e-58
4.070383e+00	2.256261e-58
4.378232e+00	2.021936e-58
4.709363e+00	1.807825e-58
5.065539e+00	1.613326e-58
5.448653e+00	1.437567e-58
5.860742e+00	1.279473e-58
6.303998e+00	1.137827e-58
6.780778e+00	1.011329e-58
7.293618e+00	8.986429e-59
7.845244e+00	7.984388e-59
8.438591e+00	7.094316e-59
9.076814e+00	6.304067e-59
9.763306e+00	5.602401e-59
1.050172e+01	4.979100e-59
1.129598e+01	4.425006e-59
1.215031e+01	3.932011e-59
1.306925e+01	3.492997e-59
1.405770e+01	3.101751e-59
1.512090e+01	2.752868e-59
1.626452e+01	2.441645e-59
1.749462e+01	2.163981e-59
1.881777e+01	1.916291e-59
2.024098e+01	1.695421e-59
2.177183e+01	1.498588e-59
2.341847e+01	1.323318e-59
2.518964e+01	1.167410e-59
2.709477e+01	1.028889e-59
2.914398e+01	9.059858e-60
3.134818e+01	7.971066e-60
3.371909e+01	7.008128e-60
3.626931e+01	6.158017e-60
3.901241e+01	5.408906e-60
4.196297e+01	4.750014e-60
4.513669e+01	4.171496e-60
4.855044e+01	3.664347e-60
5.222238e+01	3.220335e-60
5.617203e+01	2.831959e-60
6.042040e+01	2.492413e-60
6.499007e+01	2.195570e-60
6.990536e+01	1.935958e-60
7.519240e+01	1.708732e-60
8.087931e+01	1.509644e-60
8.699632e+01	1.334996e-60
9.357598e+01	1.181586e-60
1.006533e+02	1.046657e-60
1.082658e+02	9.278348e-61
1.164541e+02	8.230721e-61
1.252617e+02	7.306016e-61
1.347354e+02	6.488909e-61
1.449256e+02	5.766076e-61
1.558865e+02	5.125903e-61
1.676765e+02	4.558255e-61
1.803581e+02	4.054293e-61
1.939988e+02	3.606318e-61
2.086712e+02	3.207634e-61
2.244533e+02	2.852432e-61
2.414290e+02	2.535673e-61
2.596886e+02	2.252990e-61
2.793292e+02	2.000595e-61
3.004553e+02	1.775187e-61
3.231791e+02	1.573886e-61
3.476216e+02	1.394162e-61
3.739127e+02	1.233780e-61
4.021923e+02	1.090758e-61
4.326106e+02	9.633272e-62
4.653296e+02	8.499009e-62
5.005231e+02	7.490514e-62
5.383783e+02	6.594896e-62
5.790966e+02	5.800497e-62
6.228945e+02	5.096761e-62
6.700049e+02	4.474127e-62
7.206783e+02	3.923938e-62
7.751842e+02	3.438365e-62
8.338125e+02	3.010334e-62
8.968749e+02	2.633465e-62
9.647068e+02	2.302016e-62
1.037669e+03	2.010825e-62
1.116149e+03	1.755265e-62
1.200565e+03	1.531194e-62
1.291366e+03	1.334914e-62
1.389033e+03	1.163129e-62
1.494088e+03	1.012905e-62
1.607088e+03	8.816375e-63
1.728634e+03	7.670190e-63
1.859373e+03	6.670059e-63
2.000000e+03	5.797933e-63
"""


def parse_center(center_string):
    parts = [float(x.strip()) for x in center_string.split(",")]
    if len(parts) != 3:
        raise ValueError("--center must have exactly 3 values: X,Y,Z")
    return tuple(parts)


def parse_box(box_string):
    parts = [float(x.strip()) for x in box_string.split(",")]
    if len(parts) != 6:
        raise ValueError("--box must have exactly 6 values: MINX,MAXX,MINY,MAXY,MINZ,MAXZ")
    return tuple(parts)


def parse_pair(value_string, cast_type=float, option_name="value"):
    parts = [cast_type(x.strip()) for x in value_string.split(",")]
    if len(parts) == 1:
        return parts[0], parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"{option_name} must have either 1 or 2 comma-separated values")


def parse_instrument(instr_string):
    parts = [x.strip() for x in instr_string.split(",")]

    axis_map = {
        "z": {"inclination": 0.0, "azimuth": 0.0, "roll": 90.0},
        "x": {"inclination": 90.0, "azimuth": 0.0, "roll": 0.0},
        "y": {"inclination": 90.0, "azimuth": -90.0, "roll": 0.0},
    }

    if len(parts) == 1 and parts[0].lower() in axis_map:
        return axis_map[parts[0].lower()].copy()

    if len(parts) == 3:
        return {
            "inclination": float(parts[0]),
            "azimuth": float(parts[1]),
            "roll": float(parts[2]),
        }

    raise ValueError(
        "--instrument must be one of:\n"
        "  x\n"
        "  y\n"
        "  z\n"
        "  INC,AZ,ROLL"
    )


def default_instruments():
    return [
        parse_instrument("z"),
        parse_instrument("x"),
        parse_instrument("y"),
    ]


def instrument_name(index):
    default_names = ["i_z", "i_x", "i_y"]
    if index < len(default_names):
        return default_names[index]
    return f"i_{index + 1}"


def parse_instrument_wavelength_grid(grid_string):
    parts = [x.strip() for x in grid_string.split(",")]

    if len(parts) == 1:
        return {
            "type": "single",
            "wavelength": float(parts[0]),
        }

    if len(parts) == 3:
        return {
            "type": "log",
            "min": float(parts[0]),
            "max": float(parts[1]),
            "num": int(parts[2]),
        }

    raise ValueError(
        "--instrument_wavelength_grid must be either:\n"
        "  WAVELENGTH\n"
        "  MIN,MAX,NUM"
    )


def convert_snapshot_to_skirt_text(snapshot_path, output_stars_path, output_gas_path):
    with h5py.File(snapshot_path, 'r') as f:

        time = f['Header'].attrs['Time'] * code_time_to_Myrs

        if 'PartType5' in f:
            stars = f['PartType5']
            star_coords = stars['Coordinates'][:]
            star_velocities = stars['Velocities'][:] / 1e3
            star_masses = stars['Masses'][:]
            star_metallicities = stars['Metallicity'][:]
            star_ages = stars['StellarFormationTime'][:] * code_time_to_Myrs
            smoothing_length_star = np.full(len(star_coords), 0.1)

            star_ages = time - star_ages

            with open(output_stars_path, 'w') as fstar:
                fstar.write("# SKIRT particle source file for stars\n")
                fstar.write("# Column 1: position x (pc)\n")
                fstar.write("# Column 2: position y (pc)\n")
                fstar.write("# Column 3: position z (pc)\n")
                fstar.write("# Column 4: smoothing length (pc)\n")
                fstar.write("# Column 5: velocity vx (km/s)\n")
                fstar.write("# Column 6: velocity vy (km/s)\n")
                fstar.write("# Column 7: velocity vz (km/s)\n")
                fstar.write("# Column 8: mass (Msun)\n")
                fstar.write("# Column 9: metallicity (1)\n")
                fstar.write("# Column 10: age (Myr)\n")
                for i in range(len(star_coords)):
                    fstar.write(
                        f"{float(star_coords[i, 0]):.6f} {float(star_coords[i, 1]):.6f} {float(star_coords[i, 2]):.6f} "
                        f"0.01 "
                        f"{float(star_velocities[i, 0] / 1e3):.6f} {float(star_velocities[i, 1] / 1e3):.6f} {float(star_velocities[i, 2] / 1e3):.6f} "
                        f"{float(star_masses[i]):.6e} 0.0142 {float(star_ages[i]):.6f}\n"
                    )

                print(f"✅ Wrote stars file: {output_stars_path}")
        else:
            print("No stars in the snapshot. Skipping stars file creation.")
            output_stars_path = None

        gas = f['PartType0']
        gas_coords = gas['Coordinates'][:]
        gas_smoothing = gas['SmoothingLength'][:]
        gas_masses = gas['Masses'][:]
        gas_metallicities = gas['Metallicity'][:]
        gas_temperatures = gas['Temperature'][:]
        gas_velocities = gas['Velocities'][:] / 1e3

        with open(output_gas_path, 'w') as fgas:
            fgas.write("# SKIRT particle medium file for gas\n")
            fgas.write("# Column 1: position x (pc)\n")
            fgas.write("# Column 2: position y (pc)\n")
            fgas.write("# Column 3: position z (pc)\n")
            fgas.write("# Column 4: smoothing length (pc)\n")
            fgas.write("# Column 5: gas mass (Msun)\n")
            fgas.write("# Column 6: metallicity (1)\n")
            fgas.write("# Column 7: temperature (K)\n")
            fgas.write("# Column 8: velocity vx (km/s)\n")
            fgas.write("# Column 9: velocity vy (km/s)\n")
            fgas.write("# Column 10: velocity vz (km/s)\n")
            for i in range(len(gas_coords)):
                fgas.write(
                    f"{gas_coords[i,0]:.6f} {gas_coords[i,1]:.6f} {gas_coords[i,2]:.6f} "
                    f"{gas_smoothing[i]:.6f} {gas_masses[i]:.6e} 0.0142 "
                    f"{gas_temperatures[i]:.2f} "
                    f"{gas_velocities[i,0]:.6f} {gas_velocities[i,1]:.6f} {gas_velocities[i,2]:.6f}\n"
                )

    print(f"✅ Wrote gas file: {output_gas_path}")


def snapshot_has_stars(snapshot_path):
    with h5py.File(snapshot_path, 'r') as h5:
        return 'PartType5' in h5


def write_isrf_file(output_path):
    with open(output_path, 'w') as f:
        f.write(ISRF_SED_DATA)
    print(f"✅ Wrote ISRF SED file: {output_path}")


def make_instrument_wavelength_block(instrument_wavelength_grid):
    if instrument_wavelength_grid["type"] == "single":
        return f"""
                        <wavelengthGrid type="WavelengthGrid">
                            <ListWavelengthGrid wavelengths="{instrument_wavelength_grid['wavelength']} micron"/>
                        </wavelengthGrid>"""

    return f"""
                        <wavelengthGrid type="WavelengthGrid">
                            <LogWavelengthGrid minWavelength="{instrument_wavelength_grid['min']} micron" maxWavelength="{instrument_wavelength_grid['max']} micron" numWavelengths="{instrument_wavelength_grid['num']}"/>
                        </wavelengthGrid>"""


def make_instrument_block(instruments, distance_mpc, center, fov, resolution, instrument_wavelength_grid):
    center_x, center_y, center_z = center
    fov_x, fov_y = fov
    num_pixels_x, num_pixels_y = resolution

    wavelength_block = make_instrument_wavelength_block(instrument_wavelength_grid)

    instrument_blocks = []
    for i, instr in enumerate(instruments):
        name = instrument_name(i)

        block = f"""
                    <FullInstrument instrumentName="{name}" distance="{distance_mpc} Mpc" inclination="{instr['inclination']} deg" azimuth="{instr['azimuth']} deg" roll="{instr['roll']} deg" fieldOfViewX="{fov_x} pc" numPixelsX="{num_pixels_x}" centerX="{center_x} pc" fieldOfViewY="{fov_y} pc" numPixelsY="{num_pixels_y}" centerY="{center_y} pc" recordComponents="false" numScatteringLevels="0" recordPolarization="false" recordStatistics="false">{wavelength_block}
                    </FullInstrument>"""
        instrument_blocks.append(block)

    return "\n".join(instrument_blocks)


def write_ski_file(
    output_ski_path,
    has_stars,
    stars_filename,
    gas_filename,
    isrf_filename,
    instruments,
    distance_mpc,
    center,
    box_bounds,
    fov,
    resolution,
    num_sites,
    num_packets,
    instrument_wavelength_grid,
):
    center_x, center_y, center_z = center
    min_x, max_x, min_y, max_y, min_z, max_z = box_bounds
    box_size_x = max_x - min_x
    box_size_y = max_y - min_y
    box_size_z = max_z - min_z
    background_radius = max(box_size_x, box_size_y, box_size_z)

    instrument_block = make_instrument_block(
        instruments=instruments,
        distance_mpc=distance_mpc,
        center=center,
        fov=fov,
        resolution=resolution,
        instrument_wavelength_grid=instrument_wavelength_grid,
    )

    if has_stars:
        source_block = f"""
                <SphericalBackgroundSource centerX="{center_x} pc" centerY="{center_y} pc" centerZ="{center_z} pc" backgroundRadius="{background_radius} pc" velocityX="0 km/s" velocityY="0 km/s" velocityZ="0 km/s" sourceWeight="1" wavelengthBias="0.5">
                    <sed type="SED">
                        <FileSED filename="{isrf_filename}"/>
                    </sed>
                    <normalization type="LuminosityNormalization">
                        <IntegratedLuminosityNormalization wavelengthRange="Source" minWavelength="0.001 micron" maxWavelength="2000 micron" integratedLuminosity="25600 Lsun"/>
                    </normalization>
                    <wavelengthBiasDistribution type="WavelengthDistribution">
                        <DefaultWavelengthDistribution/>
                    </wavelengthBiasDistribution>
                </SphericalBackgroundSource>
                <ParticleSource filename="{stars_filename}" importVelocity="true" importVelocityDispersion="false" importCurrentMass="false" importBias="false" useColumns="" sourceWeight="1" wavelengthBias="0.5">
                    <smoothingKernel type="SmoothingKernel">
                        <CubicSplineSmoothingKernel/>
                    </smoothingKernel>
                    <sedFamily type="SEDFamily">
                        <BruzualCharlotSEDFamily imf="Salpeter" resolution="High"/>
                    </sedFamily>
                    <wavelengthBiasDistribution type="WavelengthDistribution">
                        <DefaultWavelengthDistribution/>
                    </wavelengthBiasDistribution>
                </ParticleSource>"""
    else:
        source_block = f"""
                <SphericalBackgroundSource centerX="{center_x} pc" centerY="{center_y} pc" centerZ="{center_z} pc" backgroundRadius="{background_radius} pc" velocityX="0 km/s" velocityY="0 km/s" velocityZ="0 km/s" sourceWeight="1" wavelengthBias="0.5">
                    <sed type="SED">
                        <FileSED filename="{isrf_filename}"/>
                    </sed>
                    <normalization type="LuminosityNormalization">
                        <IntegratedLuminosityNormalization wavelengthRange="Source" minWavelength="0.001 micron" maxWavelength="2000 micron" integratedLuminosity="25600 Lsun"/>
                    </normalization>
                    <wavelengthBiasDistribution type="WavelengthDistribution">
                        <DefaultWavelengthDistribution/>
                    </wavelengthBiasDistribution>
                </SphericalBackgroundSource>"""

    ski_text = f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- A SKIRT parameter file © Astronomical Observatory, Ghent University -->
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9" producer="SKIRT v9.0 (git 1cb67c4 built on 09/07/2025 at 14:31:18)" time="2025-07-16T14:00:08.966">
    <MonteCarloSimulation userLevel="Expert" simulationMode="DustEmission" iteratePrimaryEmission="true" iterateSecondaryEmission="true" numPackets="{num_packets}">
        <random type="Random">
            <Random seed="42"/>
        </random>
        <units type="Units">
            <ExtragalacticUnits wavelengthOutputStyle="Wavelength" fluxOutputStyle="Frequency"/>
        </units>
        <cosmology type="Cosmology">
            <LocalUniverseCosmology/>
        </cosmology>
        <sourceSystem type="SourceSystem">
            <SourceSystem minWavelength="0.001 micron" maxWavelength="100 micron" wavelengths="0.55 micron" sourceBias="0.5">
                <sources type="Source">{source_block}
                </sources>
            </SourceSystem>
        </sourceSystem>
        <mediumSystem type="MediumSystem">
            <MediumSystem>
                <photonPacketOptions type="PhotonPacketOptions">
                    <PhotonPacketOptions explicitAbsorption="false" forceScattering="true" minWeightReduction="1e4" minScattEvents="0" pathLengthBias="0.5"/>
                </photonPacketOptions>
                <dynamicStateOptions type="DynamicStateOptions">
                    <DynamicStateOptions>
                        <recipes type="DynamicStateRecipe">
                            <LinearDustDestructionRecipe maxNotConvergedCells="0" densityFractionTolerance="0.05" minSilicateTemperature="1200 K" maxSilicateTemperature="1200 K" minGraphiteTemperature="2000 K" maxGraphiteTemperature="2000 K"/>
                        </recipes>
                    </DynamicStateOptions>
                </dynamicStateOptions>
                <radiationFieldOptions type="RadiationFieldOptions">
                    <RadiationFieldOptions storeRadiationField="true">
                        <radiationFieldWLG type="DisjointWavelengthGrid">
                            <LogWavelengthGrid minWavelength="0.1 micron" maxWavelength="2000 micron" numWavelengths="200"/>
                        </radiationFieldWLG>
                    </RadiationFieldOptions>
                </radiationFieldOptions>
                <secondaryEmissionOptions type="SecondaryEmissionOptions">
                    <SecondaryEmissionOptions storeEmissionRadiationField="false" secondaryPacketsMultiplier="1" spatialBias="0.5" sourceBias="0.5"/>
                </secondaryEmissionOptions>
                <iterationOptions type="IterationOptions">
                    <IterationOptions minPrimaryIterations="1" maxPrimaryIterations="10" minSecondaryIterations="1" maxSecondaryIterations="10" includePrimaryEmission="false" primaryIterationPacketsMultiplier="1" secondaryIterationPacketsMultiplier="1"/>
                </iterationOptions>
                <dustEmissionOptions type="DustEmissionOptions">
                    <DustEmissionOptions dustEmissionType="Stochastic" includeHeatingByCMB="true" maxFractionOfPrimary="0.01" maxFractionOfPrevious="0.03" sourceWeight="1" wavelengthBias="0.5">
                        <cellLibrary type="SpatialCellLibrary">
                            <AllCellsLibrary/>
                        </cellLibrary>
                        <dustEmissionWLG type="DisjointWavelengthGrid">
                            <LogWavelengthGrid minWavelength="0.1 micron" maxWavelength="2000 micron" numWavelengths="200"/>
                        </dustEmissionWLG>
                        <wavelengthBiasDistribution type="WavelengthDistribution">
                            <DefaultWavelengthDistribution/>
                        </wavelengthBiasDistribution>
                    </DustEmissionOptions>
                </dustEmissionOptions>
                <media type="Medium">
                    <ParticleMedium filename="{gas_filename}" massType="Mass" massFraction="1" importMetallicity="true" importTemperature="true" maxTemperature="0 K" importVelocity="true" importMagneticField="false" importVariableMixParams="false" useColumns="">
                        <smoothingKernel type="SmoothingKernel">
                            <CubicSplineSmoothingKernel/>
                        </smoothingKernel>
                        <materialMix type="MaterialMix">
                            <DraineLiDustMix numSilicateSizes="5" numGraphiteSizes="5" numPAHSizes="5"/>
                        </materialMix>
                    </ParticleMedium>
                </media>
                <samplingOptions type="SamplingOptions">
                    <SamplingOptions numDensitySamples="100" numPropertySamples="1" aggregateVelocity="Average"/>
                </samplingOptions>
                <grid type="SpatialGrid">
                    <VoronoiMeshSpatialGrid minX="{min_x} pc" maxX="{max_x} pc" minY="{min_y} pc" maxY="{max_y} pc" minZ="{min_z} pc" maxZ="{max_z} pc" policy="ImportedSites" numSites="{num_sites}" filename="" relaxSites="false"/>
                </grid>
            </MediumSystem>
        </mediumSystem>
        <instrumentSystem type="InstrumentSystem">
            <InstrumentSystem>
                <defaultWavelengthGrid type="WavelengthGrid">
                    <LogWavelengthGrid minWavelength="0.1 micron" maxWavelength="2000 micron" numWavelengths="200"/>
                </defaultWavelengthGrid>
                <instruments type="Instrument">
{instrument_block}
                </instruments>
            </InstrumentSystem>
        </instrumentSystem>
        <probeSystem type="ProbeSystem">
            <ProbeSystem>
                <probes type="Probe">
                    <DustEmissionWavelengthGridProbe probeName="dust_emission"/>
                    <ConvergenceInfoProbe probeName="cns" wavelength="0.55 micron" probeAfter="Run"/>
                </probes>
            </ProbeSystem>
        </probeSystem>
    </MonteCarloSimulation>
</skirt-simulation-hierarchy>
"""

    with open(output_ski_path, 'w') as f:
        f.write(ski_text)

    print(f"✅ Wrote SKIRT parameter file: {output_ski_path}")


def run_skirt(ski_path, output_path, mpi_exe="mpirun", mpi_processes=1, threads=1):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["SKIRT_NUM_THREADS"] = str(threads)

    ski_filename = Path(ski_path).name

    if mpi_processes > 1:
        cmd = [
            mpi_exe,
            "-np",
            str(mpi_processes),
            "skirt",
            "-t",
            str(threads),
            ski_filename,
        ]
        print(f"Starting SKIRT with {mpi_processes} MPI processes and {threads} threads each")
    else:
        cmd = [
            "skirt",
            "-t",
            str(threads),
            ski_filename,
        ]
        print(f"Starting SKIRT with 1 MPI process and {threads} threads")

    print("Running command:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(output_path),
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"SKIRT failed with return code {result.returncode}")

    print("✅ SKIRT run completed successfully.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create self-contained SKIRT input files from a simulation snapshot and run SKIRT.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("snapshot", help="Path to HDF5 snapshot file")

    parser.add_argument(
        "--output_path",
        default="skirt_output",
        help="Output directory for generated files and SKIRT outputs [default: ./skirt_output]",
    )
    parser.add_argument("--stars", default="stars.txt", help="Output stars text filename")
    parser.add_argument("--gas", default="gas.txt", help="Output gas text filename")
    parser.add_argument("--ski", default="run.ski", help="Output SKIRT parameter filename")

    parser.add_argument("--center", default=None, help="Center X,Y,Z in pc")
    parser.add_argument("--box", default="0,100,0,100,0,100", help="Box bounds in pc")
    parser.add_argument("--fov", default="25,25", help="Field of view X[,Y] in pc")
    parser.add_argument("--resolution", default="1024,1024", help="Resolution N[,M]")
    parser.add_argument("--distance", type=float, default=0.0005, help="Distance in Mpc")
    parser.add_argument("--num_sites", type=int, default=500000, help="Number of Voronoi sites")
    parser.add_argument("--num_packets", default="5e7", help="Number of photon packets")

    parser.add_argument(
        "--instrument",
        action="append",
        default=None,
        help=(
            "Instrument orientation. Can be repeated. Allowed forms:\n"
            "  x\n"
            "  y\n"
            "  z\n"
            "  INC,AZ,ROLL"
        ),
    )

    parser.add_argument(
        "--instrument_wavelength_grid",
        default="0.1,2000,200",
        help=(
            "Shared wavelength grid for all instruments. Allowed forms:\n"
            "  WAVELENGTH\n"
            "  MIN,MAX,NUM\n"
            "Default: 0.1,2000,200"
        ),
    )

    parser.add_argument("--mpi_processes", type=int, default=1, help="Number of MPI processes")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads per MPI process")
    parser.add_argument("--mpi_exe", default="mpirun", help=argparse.SUPPRESS)

    parser.add_argument(
        "--no_run_skirt",
        action="store_true",
        help="Only prepare files; do not run SKIRT",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    snapshot_path = str(Path(args.snapshot).resolve())
    stars_path = str(output_path / args.stars)
    gas_path = str(output_path / args.gas)
    ski_path = str(output_path / args.ski)
    isrf_path = str(output_path / "ISRF_sed.dat")

    box_bounds = parse_box(args.box)
    min_x, max_x, min_y, max_y, min_z, max_z = box_bounds

    if args.center is None:
        center = (
            0.5 * (min_x + max_x),
            0.5 * (min_y + max_y),
            0.5 * (min_z + max_z),
        )
    else:
        center = parse_center(args.center)

    fov = parse_pair(args.fov, cast_type=float, option_name="--fov")
    resolution = parse_pair(args.resolution, cast_type=int, option_name="--resolution")

    if args.instrument is None:
        instruments = default_instruments()
    else:
        instruments = [parse_instrument(instr) for instr in args.instrument]

    instrument_wavelength_grid = parse_instrument_wavelength_grid(args.instrument_wavelength_grid)

    has_stars = snapshot_has_stars(snapshot_path)
    print(f"Snapshot has stars: {has_stars}")
    print(f"Output path: {output_path}")
    print(f"Configured SKIRT parallelism: mpi_processes={args.mpi_processes}, threads={args.threads}")

    convert_snapshot_to_skirt_text(snapshot_path, stars_path, gas_path)
    write_isrf_file(isrf_path)

    write_ski_file(
        output_ski_path=ski_path,
        has_stars=has_stars,
        stars_filename=args.stars,
        gas_filename=args.gas,
        isrf_filename="ISRF_sed.dat",
        instruments=instruments,
        distance_mpc=args.distance,
        center=center,
        box_bounds=box_bounds,
        fov=fov,
        resolution=resolution,
        num_sites=args.num_sites,
        num_packets=args.num_packets,
        instrument_wavelength_grid=instrument_wavelength_grid,
    )

    if not args.no_run_skirt:
        run_skirt(
            ski_path=ski_path,
            output_path=output_path,
            mpi_exe=args.mpi_exe,
            mpi_processes=args.mpi_processes,
            threads=args.threads,
        )
    else:
        print("✅ Finished preparing SKIRT input files. SKIRT was not run.")


if __name__ == "__main__":
    main()