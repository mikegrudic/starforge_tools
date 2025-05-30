Black hole-GMC encounter setup
------------------------------

|gmc_bh.mp4|

Cloning the repo
~~~~~~~~~~~~~~~~

First get the code via <WRAP prewrap>

.. code:: bash

   git clone https://bitbucket.org/guszejnov/gizmo_imf.git

</WRAP>

Compiling the code
~~~~~~~~~~~~~~~~~~

Many physics modules and options are specified at compile-time in GIZMO, via C precompiler flags listed in a file Config.sh that you must create. See the `gizmo documentation <http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html>`__ for some explanation of what the different flags do.

Here we provide some example ``Config.sh`` files for different physics setups for simulating a black hole-GMC encounter.

Notes on OPENMP and MULTIPLEDOMAINS flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that all setups may optionally include the ``OPENMP`` and ``MULTIPLEDOMAINS`` flags for performance tuning - you will need to experiment to determine the optimal settings for your setup and machine - see the `GIZMO documentation <http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html>`__ for details on what these do. For medium sized (~2e7 gas cells) runs on Frontera or Stampede-2, ``OPENMP=2`` and leaving out ``MULTIPLEDOMAINS`` (which defaults to 8) work well as a starting point.

Config.sh for a minimal MHD Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This setup has no thermochemical ISM physics, just uniformly-cold, 10K, magnetized gas. It can be considered a \``bare-minimum” setup for modeling the most basic MHD dynamics of a GMC, but will produce far too many stars because there is no feedback to regulate star formation.

<WRAP prewrap>

.. code:: bash

   # MHD/plasma physics
   MAGNETIC # enables MHD
   CONDUCTION           # Condution solver
   CONDUCTION_SPITZER   # Spitzer conduction
   VISCOSITY            # Viscosity solver
   VISCOSITY_BRAGINSKII # Braginskii viscosity
   DIFFUSION_OPTIMIZERS # optimization flags

   # star formation criteria 
   GALSF # enables star formation
   GALSF_SFR_VIRIAL_SF_CRITERION=0 # standard FIRE-3 virial criterion
   GALSF_SFR_TIDAL_HILL_CRITERION # tidal criterion
   GALSF_SFR_IMF_SAMPLING # we sample discrete O stars for more realistic feedback vs. IMF-averaged
   SINGLE_STAR_AND_SSP_HYBRID_MODEL=0.01 # Type 5 star particles form if mass resolution is smaller than this; otherwise Type 4 star cluster particles

   # sink particle/black hole settings
   SINGLE_STAR_TIMESTEPPING=0
   BLACK_HOLES
   BH_CALC_DISTANCES
   SINGLE_STAR_SINK_DYNAMICS
   BH_SWALLOWGAS
   BH_GRAVCAPTURE_GAS
   BH_GRAVCAPTURE_FIXEDSINKRADIUS

   # Gravity/integration settings
   RT_ISRF_BACKGROUND=1
   ADAPTIVE_GRAVSOFT_FORGAS
   TIDAL_TIMESTEP_CRITERION
   ADAPTIVE_TREEFORCE_UPDATE=0.125
   GRAVITY_ACCURATE_FEWBODY_INTEGRATION
   SINGLE_STAR_FB_TIMESTEPLIMIT

   # Boundary conditions
   BOX_PERIODIC
   GRAVITY_NOT_PERIODIC

   # FIRE-3 ISM physics + feedback model
   FIRE_PHYSICS_DEFAULTS=3

</WRAP>

Config.sh for full ISM physics setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This setup includes all of the prescriptions for star formation, feedback, and ISM physics as described in the `FIRE-3 <https://arxiv.org/abs/2203.00040v1>`__ paper.

<WRAP prewrap>

.. code:: bash

   # MHD/plasma physics
   MAGNETIC # enables MHD
   CONDUCTION           # Condution solver
   CONDUCTION_SPITZER   # Spitzer conduction
   VISCOSITY            # Viscosity solver
   VISCOSITY_BRAGINSKII # Braginskii viscosity
   DIFFUSION_OPTIMIZERS # optimization flags

   # star formation criteria 
   GALSF # enables star formation
   GALSF_SFR_VIRIAL_SF_CRITERION=0 # standard FIRE-3 virial criterion
   GALSF_SFR_TIDAL_HILL_CRITERION # tidal criterion
   GALSF_SFR_IMF_SAMPLING # we sample discrete O stars for more realistic feedback vs. IMF-averaged
   SINGLE_STAR_AND_SSP_HYBRID_MODEL=0.01 # Type 5 star particles form if mass resolution is smaller than this; otherwise Type 4 star cluster particles

   # sink particle/black hole settings
   SINGLE_STAR_TIMESTEPPING=0
   BLACK_HOLES
   BH_CALC_DISTANCES
   SINGLE_STAR_SINK_DYNAMICS
   BH_SWALLOWGAS
   BH_GRAVCAPTURE_GAS
   BH_GRAVCAPTURE_FIXEDSINKRADIUS

   # Gravity/integration settings
   RT_ISRF_BACKGROUND=1
   ADAPTIVE_GRAVSOFT_FORGAS
   TIDAL_TIMESTEP_CRITERION
   ADAPTIVE_TREEFORCE_UPDATE=0.125
   GRAVITY_ACCURATE_FEWBODY_INTEGRATION
   SINGLE_STAR_FB_TIMESTEPLIMIT

   # Boundary conditions
   BOX_PERIODIC
   GRAVITY_NOT_PERIODIC

   # FIRE-3 ISM physics + feedback model
   FIRE_PHYSICS_DEFAULTS=3

</WRAP>

Once you have put your desired flags in your Config.sh, make sure your machine is the one uncommented in ``Makefile.systype``, then run the ``make`` command. This generates the ``GIZMO`` binary.

Initial conditions
~~~~~~~~~~~~~~~~~~

Turbulent GMC initial conditions can be generated with the code `MakeCloud <https://github.com/mikegrudic/MakeCloud>`__. Clone the repo and link MakeCloud.py to a directory in your path so you can use from the command line. The command line interface options can be listed by running ``MakeCloud.py -h``. As an example, to generate a 2e6msun, 100pc cloud and a 1e5msun black hole 500pc away moving toward it at 30km/s, with a mass resolution of 10msun, specifying that we want to use km/s as our unit velocity and Gauss as our unit magnetic field, we would do: <WRAP prewrap>

.. code:: bash

   MakeCloud.py  --M=2e6 --R=100 --N=200000 --Mstar=1e5 --v_star=30,0,00 --x_star=0,500,500 --v_unit=1000 --B_unit=1

</WRAP>

This will generate an HDF5 initial conditions file and a parameters file containing the runtime parameters for the run.

Running the code
~~~~~~~~~~~~~~~~

Once you have your GIZMO binary, all required datafiles (`spcool_tables <https://bitbucket.org/phopkins/gizmo-public/downloads/spcool_tables.tgz>`__ and `TREECOOL <https://bitbucket.org/phopkins/gizmo-public/downloads/TREECOOL.txt>`__), and your parameters file in your simulation directory, you are ready to run the simulation. If you are working on a cluster, this will generally involve submitting a batch job with the command to run the code. An example submit script using the SLURM queuing system on Frontera is

<WRAP prewrap>

.. code:: bash

   #!/bin/bash                                                                                                                                             
   #SBATCH -J M2e6_R100_Z1_S0_A2_B0.1_I1_Res58_n2_sol0.5_42 -p development -N 1 --ntasks-per-node 56 -t 01:00:00 -A AST21002                               
   source $HOME/.bashrc
   ibrun ./GIZMO ./params_M2e6_Mstar100000_R100_Z1_S0_A2_B0.1_I1_Res58_n2_sol0.5_42.txt 0 1>gizmo.out 2>gizmo.err &
   wait

</WRAP>

which one would submit to the queue via the command ``sbatch myscript.sh``. Note the flag ``0`` after the path of the params file: this flag specifies whether the code is to start a brand new simulation from the IC (0), restart an already-existing simulation with checkpoint files (1), or restart from a snapshot (2).

Visualization
~~~~~~~~~~~~~

There are many tools for visualizing GIZMO simulations. The most widely-used is probably yt. A useful rendering backend is `meshoid <https://github.com/mikegrudic/meshoid>`__, and the quickstart shows you how to make vaguely publish-worthy colormap plot of the data.

The STARFORGE collaboration has also developed a python package `CrunchSnaps <https://github.com/mikegrudic/CrunchSnaps>`__ originally for doing large rendering runs like `this <https://www.youtube.com/watch?v=LeX5e51UkzI>`__, but the package also contains the script ``SinkVis2.py`` that implements a simple command-line interface for making quick renders of the simulation data. Once the package is installed and ``SinkVis2.py`` is in your path, you can do a simple rendering run on a simulation via

<WRAP prewrap>

.. code:: bash

   SinkVis2.py snapshot*.hdf5

</WRAP>

See ``SinkVis2.py -h`` for detailed options.

.. |gmc_bh.mp4| image:: /gmc_bh.mp4