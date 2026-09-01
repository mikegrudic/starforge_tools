Getting set up on TACC systems
------------------------------

These instructions cover getting set up to run STARFORGE simulations on TACC systems, using Frontera in the examples. For other TACC machines (e.g. Stampede3, Vista), substitute the appropriate hostname, core counts, and queue names — see the machine's `user guide <https://docs.tacc.utexas.edu>`__.

Initial login/bash/environment setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Request a TACC account `here <https://accounts.tacc.utexas.edu/register>`__ and wait for approval/confirmation.
- Give your TACC username to your collaborator who manages the allocation, so that they can give you access.
- Set up TACC multi-factor authentication
- If you have not set up your SSH keys, do so
- Login:

  .. code:: bash

     ssh -XY username@frontera.tacc.utexas.edu

- Note: logging in will take you to the “head node”: a special node in the cluster that is shared between users for accessing the system, managing jobs, and running small, short analysis and data management tasks. Don’t run anything that requires more than ~a minute to run on the head node, and never run a multi-core job or simulation. For heavy computing, either submit a job (to run automatically, see example script below) or get an interactive node if you just want to run stuff in the terminal: ``idev -A <allocation code> -m <how many minutes you want to use a node for>``. Note that interactive sessions are charged to the allocation at the same rate per node-hour as batch jobs. Note that for short (<2 hr) interactive jobs it is good practice to submit to the development queue as it queues much faster, to do that add ``-p development`` to your idev call.
- Download and install your personal python setup, since you can’t manage your own packages with TACC’s python module. A good option is Miniforge:

  .. code:: bash

     wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
     bash Miniforge3-Linux-x86_64.sh

- Edit the file .bashrc in your home directory, adding the lines in the block below. This is needed because loading the module impi overwrites PYTHONPATH to TACC’s Intel Python environment, whose packages will override anything you have on your setup (replace ``python3.X`` with the python version you installed):

  .. code:: bash

     umask 022
     ulimit -s unlimited
     module load TACC intel impi hdf5 gsl fftw2
     export PATH=$HOME/miniforge3/bin:$PATH
     export PYTHONPATH=$HOME/miniforge3/lib/python3.X/site-packages

- You can also add any personalized macros here, e.g. ``alias nemacs='emacs -nw'`` to quickly open emacs in terminal mode
- Run ``source ~/.bashrc`` to update your bash settings to include the stuff you just added - this only needs to be done whenever you modify your .bashrc
- Double-check that the python version mentioned in the startup message of python is the one you just tried to install. If not, check that your paths are set up correctly in your .bashrc
- To install any extra python packages you may need that are on PyPI, run ``pip install <package name>``

Getting and compiling GIZMO
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Go to your work/scratch directory: ``cd $WORK`` (note that $WORK has finite storage space, ~1TB, so it should fit all but the largest simulations. If you need more, use $SCRATCH. IMPORTANT: scratch filesystems get periodically purged, any file that is not accessed in the last few weeks might be deleted.)
- Clone the gizmo repo (see :doc:`getting_the_code` for the code's use policies):

  .. code:: bash

     git clone -b starforge_dev https://github.com/pfhopkins/gizmo.git

- Enter the gizmo directory (``cd gizmo``)
- Open the file Makefile.systype and make sure that the SYSTYPE line matching your machine (e.g. ``SYSTYPE="Frontera"``) is un-commented (and the others commented out). Note that pulling a newer code version might overwrite this, so it is recommended to check after each git pull.
- Create the file Config.sh and enter the list of compiler flags you want in it
- Build the code: ``make`` (if recompiling, precede with a ``make clean``)

You now have the compiled GIZMO binary file in the gizmo directory.

Setting up MakeCloud
~~~~~~~~~~~~~~~~~~~~

- Install `MakeCloud <https://github.com/mikegrudic/MakeCloud>`__: ``pip install makecloud``
- The command line options can be listed by running ``MakeCloud -h``. Required glass files are downloaded automatically on first use, and turbulent velocity fields are generated and cached (see ``--glass_path`` and ``--turb_path`` to control where these are stored).
- Run the script, e.g. for a 2e3msun, 3pc GMC with 2e6 gas cells surrounded by a box-filling diffuse medium (the default; disable with ``--no_diffuse_gas``):

  .. code:: bash

     MakeCloud --M=2e3 --R=3 --N=2000000

Running simulations
~~~~~~~~~~~~~~~~~~~

Let’s assume you have the GIZMO binary, initial conditions file, and params file params.txt all ready to go. A job (e.g. a simulation) is submitted to the queue like so: ``sbatch myjob.sh``

Where myjob.sh is the batch script for that job. A minimal template batch script for running GIZMO is:

.. code:: bash

   #!/bin/bash
   #SBATCH -J name_of_job -p normal -N 1 --ntasks-per-node 56 -t 48:00:00 -A <your allocation>

   source $HOME/.bashrc
   ibrun ./GIZMO ./params.txt 0 1>gizmo.out 2>gizmo.err &
   wait

But note that you can run any command in place of the ibrun command used to run gizmo above.

Key for the different options in the header:

- ``-J`` the name of the job - can be anything, but should be somewhat descriptive so you know which job is which
- ``-p`` the queue the job is being submitted to - ‘normal’ for regular jobs, but you can use ‘development’ to get a queue that will run much sooner for testing purposes, but can only run <2hr jobs one at a time.
- ``-N`` The number of nodes you want to run the job on. Each node on Frontera has 56 cores and 192GB of RAM; other machines differ, so check the user guide.
- ``--ntasks-per-node`` The number of MPI processes per node - normally equal to the number of cores per node, but if you are compiling with OPENMP=<N> then divide this number by N, and add
- ``-t`` the maximum time you want the job to run for, formatted hours:minutes:seconds. The maximum for the normal queue is 48 hours, and if your run finishes before the time is up, the job will terminate.
- ``-A`` the allocation you are charging for your CPU time

When you submit the job, it will wait in the queue for some amount of time until it eventually runs. To check on the status of all jobs you have submitted, use the command ``showq -u <your username>``. There are many other arguments you can give to showq to customize the information you get. To get rich information on a certain job, use the command ``scontrol show jobid <job id #>``
