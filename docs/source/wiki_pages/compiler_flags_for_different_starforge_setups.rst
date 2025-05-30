Compiler Flags for Different STARFORGE setups
---------------------------------------------

Here we provide some example ``Config.sh`` files for different types of STARFORGE physics setups.

Notes on OPENMP and MULTIPLEDOMAINS flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that all setups may optionally include the ``OPENMP`` and ``MULTIPLEDOMAINS`` flags for performance tuning - you will need to experiment to determine the optimal settings for your setup and machine - see the `GIZMO documentation <http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html>`__ for details on what these do. For medium sized (~2e7 gas cells) runs on Frontera or Stampede-2, ``OPENMP=2`` and leaving out ``MULTIPLEDOMAINS`` (which defaults to 8) work well as a starting point.

“Standard” STARFORGE “Full Physics” Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the setup used in most STARFORGE runs, including all important feedback mechanisms and using explicit radiative transfer in 5 bands - e.g. `as in this paper <https://arxiv.org/abs/2201.00882>`__.


.. code:: bash

   SINGLE_STAR_STARFORGE_DEFAULTS # package that enables many flags for timestepping, integration, and sink formation and accretion
   MAGNETIC # enables the MHD solver
   COOLING # enables detailed cooling physics, see the FIRE-3 paper for details
   SINGLE_STAR_FB_JETS # enables protostellar jet feedback
   SINGLE_STAR_FB_RAD # enables explicit radiative transfer, include emission/absorption of gas and dust, and stellar emission
   SINGLE_STAR_FB_WINDS=2 # enables stellar winds with eddington-limited prescription at high mass
   SINGLE_STAR_FB_SNE # enables SNe
   BOX_PERIODIC # sets a periodic domain to have a self-consisent magnetic field topology
   GRAVITY_NOT_PERIODIC # gravity is solved with vacuum boundary conditions
   ADAPTIVE_TREEFORCE_UPDATE=0.0625 # Enables gravity optimization for gas cells introduced in https://arxiv.org/abs/2010.13792, numerical value means that gravity is only updated at most every 1/16 of a tidal timestep, where the tidal timestep is given in https://arxiv.org/abs/1910.06349


Isothermal MHD Setup
~~~~~~~~~~~~~~~~~~~~

This is the simplest setup that will form stars and produce well-converged (but incorrect!) answers for the mass moments of the IMF (see `this paper <https://arxiv.org/abs/2002.01421v1>`__).


.. code:: bash

   SINGLE_STAR_STARFORGE_DEFAULTS # package that enables many flags for timestepping, integration, and sink formation and accretion
   MAGNETIC # enables the MHD solver
   EOS_ENFORCE_ADIABAT=4e4 # square of isothermal sound speed in code units
   EOS_GAMMA=1.001 # effectively isothermal EOS
   BOX_PERIODIC # sets a periodic domain to have a self-consisent magnetic field topology
   GRAVITY_NOT_PERIODIC # gravity is solved with vacuum boundary conditions
   ADAPTIVE_TREEFORCE_UPDATE=0.0625 # Enables gravity optimization for gas cells introduced in https://arxiv.org/abs/2010.13792, numerical value means that gravity is only updated at most every 1/16 of a tidal timestep, where the tidal timestep is given in https://arxiv.org/abs/1910.06349


High-redshift SIGO (pop III?) setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Includes gravity, cooling, H2 chemistry, and sink particles (but no feedback).


.. code:: bash

   SINGLE_STAR_STARFORGE_DEFAULTS # package that enables many flags for timestepping, integration, and sink formation and accretion
   COOLING # cooling function includes key coolants for pristine gas, + SINGLE_STAR_STARFORGE_DEFAULTS also enables H2 chemistry network
   BOX_PERIODIC # sets a periodic domain to have a self-consisent magnetic field topology
   GRAVITY_NOT_PERIODIC # gravity is solved with vacuum boundary conditions
   ADAPTIVE_TREEFORCE_UPDATE=0.0625 # Enables gravity optimization for gas cells introduced in https://arxiv.org/abs/2010.13792, numerical value means that gravity is only updated at most every 1/16 of a tidal timestep, where the tidal timestep is given in https://arxiv.org/abs/1910.06349
