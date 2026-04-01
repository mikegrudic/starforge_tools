# run_skirt.py

`run_skirt.py` is a self-contained script for preparing and running a SKIRT radiative transfer calculation directly from a simulation snapshot. The goal is to require only the snapshot as external input, while generating all other needed files automatically.

The script:

- reads an HDF5 simulation snapshot
- detects whether stellar particles are present
- writes a SKIRT-compatible star particle source file
- writes a SKIRT-compatible gas particle medium file
- writes an internal ISRF SED file
- builds a `.ski` file automatically
- runs SKIRT by default
- supports threaded and MPI execution from the command line

This script is designed so that it can be run locally, but the same command-line options can also be passed through a batch system such as SLURM.

---

## Overview

The script assumes that the simulation snapshot contains:

- gas in `PartType0`
- stars in `PartType5`, if present

It uses those particle datasets to produce the SKIRT input files needed for a dust-emission calculation with:

- a particle-based gas medium
- an optional particle-based stellar source component
- a spherical background ISRF source
- a configurable set of instruments
- a configurable shared instrument wavelength grid

All files are written into a single output directory, and SKIRT is run from that directory so that all generated products remain together.

---

## Main features

### Self-contained workflow

The only required external input is the simulation snapshot. The script internally generates:

- `stars.txt`
- `gas.txt`
- `ISRF_sed.dat`
- `run.ski`

and then runs SKIRT unless told not to.

### Particle detection

The script determines whether stellar particles exist by checking for `PartType5` in the snapshot. If stars are not present, the SKIRT parameter file is built without a particle stellar source.

### Configurable instruments

The script supports any number of instruments using repeated `--instrument` arguments.

Each instrument can be defined either by a shorthand axis label:

- `x`
- `y`
- `z`

or by explicit viewing angles:

- `inclination,azimuth,roll`

If no instruments are specified, the script defaults to three orthogonal views:

- `z`
- `x`
- `y`

### Shared instrument wavelength grid

All instruments share one wavelength grid, set by `--instrument_wavelength_grid`.

This can be either:

- a single wavelength, or
- a logarithmic wavelength grid defined by minimum wavelength, maximum wavelength, and number of bins

### Parallel execution support

The script supports:

- serial execution
- multithreaded execution
- MPI + multithreaded execution

The number of MPI processes and threads per process are specified on the command line.

---

## Requirements

This script requires Python 3 and the following Python packages:

- `numpy`
- `h5py`
- `astropy`

It also requires a working SKIRT installation available on your system path as:

```bash
skirt
```

## Basic usage

The script requires only a simulation snapshot as input. In its default mode, it will:

- create an output directory called `./skirt_output` if it does not already exist
- write all generated SKIRT input files into that directory
- run SKIRT immediately after preparing the files

A minimal run looks like:

```bash
python run_skirt.py snapshot.hdf5
```

If you want to generate the files without actually launching SKIRT, use:
```bash 
python run_skirt.py snapshot.hdf5 --no_run_skirt
```

If you want all files and SKIRT outputs to go somewhere other than ./skirt_output, set a custom output directory:
```bash 
python run_skirt.py snapshot.hdf5 --output_path=my_skirt_run
```



## More detailed explanation

## Tunable parameters

`run_skirt.py` is designed so that the snapshot is the only required input, but many aspects of the SKIRT setup can be adjusted from the command line.

### Output and file names

- `--output_path=DIR`  
  Directory where all generated files and SKIRT outputs will be written.  
  Default: `./skirt_output`

- `--stars=FILE`  
  Name of the exported stellar particle file.  
  Default: `stars.txt`

- `--gas=FILE`  
  Name of the exported gas particle file.  
  Default: `gas.txt`

- `--ski=FILE`  
  Name of the generated SKIRT parameter file.  
  Default: `run.ski`

### Geometry of the simulation box

- `--box=MINX,MAXX,MINY,MAXY,MINZ,MAXZ`  
  Sets the spatial bounds of the SKIRT Voronoi grid in pc.  
  Default: `0,100,0,100,0,100`

- `--center=X,Y,Z`  
  Sets the center of the image and the spherical background source in pc.  
  If not provided, the script uses the geometric center of the box.

The spherical background source radius is not set separately. It is automatically taken to be the maximum dimension of the box.

### Instrument setup

- `--instrument=SPEC`  
  Adds an instrument. This option can be repeated any number of times.

  Allowed forms:
  - `x`
  - `y`
  - `z`
  - `INC,AZ,ROLL`

  If no instruments are provided, the script defaults to three orthogonal views:
  - `z`
  - `x`
  - `y`

  The shorthand instruments correspond to:
  - `z` → inclination `0`, azimuth `0`, roll `90`
  - `x` → inclination `90`, azimuth `0`, roll `0`
  - `y` → inclination `90`, azimuth `-90`, roll `0`

- `--distance=D`  
  Distance of all instruments in Mpc.  
  Default: `0.0005`

- `--fov=X[,Y]`  
  Instrument field of view in pc. If one value is given, the same value is used for both axes.  
  Default: `25,25`

- `--resolution=N[,M]`  
  Instrument image resolution in pixels. If one value is given, the same value is used for both axes.  
  Default: `1024,1024`

- `--instrument_wavelength_grid=SPEC`  
  Shared wavelength grid used by all instruments.

  Allowed forms:
  - `WAVELENGTH`
  - `MIN,MAX,NUM`

  Examples:
  - `24` → single wavelength at `24 micron`
  - `0.1,2000,200` → logarithmic grid from `0.1` to `2000 micron` with `200` wavelengths

  Default: `0.1,2000,200`

### Monte Carlo and grid controls

- `--num_sites=N`  
  Number of Voronoi sites used in the imported-site Voronoi mesh.  
  Default: `500000`

- `--num_packets=N`  
  Number of photon packets used by SKIRT.  
  Default: `5e7`

### Parallel execution

- `--mpi_processes=N`  
  Number of MPI processes used when launching SKIRT.  
  Default: `1`

- `--threads=N`  
  Number of threads per MPI process.  
  Default: `1`

- `--no_run_skirt`  
  Prepare the input files but do not launch SKIRT.

---

## Example commands

### Use only one instrument

This produces only a single face-on z view.
```bash
python run_skirt.py snapshot.hdf5 –instrument=z
```
    
### Use custom viewing angles

This creates two instruments with user-defined orientations.
```bash
python run_skirt.py snapshot.hdf5 \
–instrument=30,45,0 \
–instrument=75,120,10
```

### Use a single-wavelength instrument output

This keeps the source setup fixed, but tells all instruments to produce output only at 24 micron.
```bash
python run_skirt.py snapshot.hdf5 \
–instrument=z \
–instrument=x \
–instrument_wavelength_grid=24
```

### Use a custom logarithmic instrument wavelength grid

```bash
python run_skirt.py snapshot.hdf5 \
–instrument=z \
–instrument_wavelength_grid=1,1000,100
```

### Increase photon packets and Voronoi sites

This may improve fidelity, but will greatly increase runtime. I fount that num_sites needs to be greater than 100000 for simi-realistic results. Improvements after 1000000 were marginal.
```bash
python run_skirt.py snapshot.hdf5 
–num_packets=1e8 
–num_sites=1000000
```

### Run with MPI and threads

This launches SKIRT with 4 MPI processes and 8 threads per process.
```bash
python run_skirt.py snapshot.hdf5 \
–mpi_processes=4 \
–threads=8
```

This corresponds to a SKIRT launch of the form:

```bash
mpirun -np 4 skirt -t 8 run.ski
```

with the environment variables:

OMP_NUM_THREADS=8
SKIRT_NUM_THREADS=8

### A realistic combined example

This example changes the box, center, instruments, instrument wavelength grid, image setup, and parallel execution all at once.

```bash
python run_skirt.py snapshot.hdf5 \
–output_path=run_001 \
–box=0,200,0,200,0,200 \  
–center=100,100,100 \
–instrument=z \
–instrument=x \
–instrument=y \
–instrument_wavelength_grid=24 \
–fov=80 \
–resolution=2048 \
–num_packets=1e8 \
–mpi_processes=4 \
–threads=8
```