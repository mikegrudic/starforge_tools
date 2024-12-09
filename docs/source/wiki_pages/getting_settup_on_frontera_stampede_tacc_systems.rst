Getting setup on Frontera/TACC systems
--------------------------------------

Initial login/bash/environment setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Request a TACC account `here <https://accounts.tacc.utexas.edu/register>`__ and wait for approval/confirmation.
- Give your TACC username to your collaborator who manages the allocation, so that they can give you access.
- Set up TACC multi-factor authentication
- If you have not set up your SSH keys, do so
- Login to Frontera:
  ::

     ssh -XY username@frontera.tacc.utexas.edu

- Note: logging in will take you to the “head node”: a special node in the cluster that is shared between users for accessing the system, managing jobs, and running small, short analysis and data management tasks. Don’t run anything that requires more than ~a minute to run on the head node, and never run a multi-core job or simulation. For heavy computing, either submit a job (to run automatically, see example script below) or get an interactive node if you just want to run stuff in the terminal: idev -A <allocation code (e.g. AST21002)> -m <how many minutes you want to use a node for>. Note that interactive sessions are charged to the allocation at the same rate per node-hour as batch jobs. Note that for short (<2 hr) interactive jobs it is good practice to submit to the development queue as it queues much faster, to do that add -p development to your idev call.
- Download and install your personal python setup, since you can’t manage your own packages with TACC’s python module. A good one is Anaconda, installed by: <code bash> wget https://repo.anaconda.com/archive/Anaconda3-2024.06-Linux-x86_64.sh

bash Anaconda3-2024.06-Linux-x86_64.sh</code>

- Edit the file .bashrc in your home directory, adding the following lines in the block below for customizations. This is needed because for some reason loading the module impi overwrites PYTHONPATH to TACC’s Intel Python environment, whose packages will override anything you have on your setup.

::

   <code bash>
     umask 022
     ulimit -s unlimited
     module load TACC intel impi hdf5 gsl fftw2
     export PATH=$HOME/anaconda3/bin:$PATH
     export PYTHONPATH=$HOME/anaconda3/lib/python3.12/site-packages
     
   </code>

::

     * You can also add any personalized macros here, e.g. I have <code bash>export SCRATCH3=/scratch3/03532/mgrudic</code> so that I can access my alternate scratch space as $SCRATCH3, and have <code bash>alias nemacs=’emacs -nw’</code> to quickly open emacs in terminal mode
   * Run source ~/.bashrc to update your bash settings to include the stuff you just added - this only needs to be done whenever you modify your .bashrc
   * Double-check that the python version mentioned in the startup message of python is the one you just tried to install. If not, check that your paths are set up correctly in your .bashrc
   * To install any extra python packages you may need that are on PyPI, run <code bash>pip install <package name></code>

Getting and compiling GIZMO
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Go to your work/scratch directory: cd $WORK (note that $WORK has finite storage space, ~1TB, so it should fit all but the largest simulations. If you need more, use $SCRATCH1. IMPORTANT: $SCRATCH1 gets periodically purged, any file that is not accessed in the last 2 weeks might be deleted.)
- Clone the gizmo repo: git clone https://bitbucket.org/guszejnov/gizmo_imf (you can also use hg clone but note that Mercurial support is ending on Bitbucket)
- Enter your bitbucket username and password, and make sure mike or david has granted you access
- Enter the gizmo directory (cd gizmo_imf)
- Open the file Makefile.systype and make sure that the line SYSTYPE=”Frontera” is un-commented (and the others commented out). Note that pulling a newer code version might overwrite this, so it is recommended to check after each git pull.
- Create the file Config.sh and enter the list of compiler flags you want in it
- Build the code: make (if recompiling, precede with a make clean)

You now have the compiled GIZMO binary file in the gizmo directory.

Setting up MakeCloud
~~~~~~~~~~~~~~~~~~~~

- From your home directory, clone the repo: cd $HOME; git clone https://github.com/mikegrudic/MakeCloud.git
- Add MakeCloud to your path so you can run it as an executable: export PATH=$HOME/MakeCloud:$PATH. You can add this command to your other custom commands in your .bashrc so it will run every time you login. Alternatively, create a directory added to your path (I use $HOME/scripts) where you keep everything you want to run like an executable, and put a symlink there: add export PATH=$HOME/scripts:$PATH to your .bashrc, and then do mkdir $HOME/scripts; cd $HOME/scripts; ln -s /path/to/MakeCloud.py .
- Download the glass file: wget http://www.tapir.caltech.edu/~mgrudich/glass_orig.npy
- If needed, edit the following lines in MakeCloud/MakeCloud.py:

  - ``--glass_path=<name>`` Contains the root path of the glass ic [default: $HOME/glass_orig.npy], replacing the path with the path to wherever you want your glass file to live.
  - ``--turb_path=<name>`` Path to store turbulent velocity fields so that we only need to generate them once [default: $HOME/turb], replacing the default path with wherever you want your stored turbulence files to live

- Run the script, e.g. for a 2e3msun, 3pc GMC with 2e6 gas cells surrounded by a box-filling medium:

::

   <code bash>  MakeCloud.py --M=2e3 --R=3 --N=2000000 --warmgas</code>

Running simulations
~~~~~~~~~~~~~~~~~~~

Let’s assume you have the GIZMO binary, initial conditions file, and params file params.txt all ready to go. A job (e.g. a simulation) is submitted to the queue like so: <code bash> sbatch myjob.sh</code>

Where myjob.sh is the batch script for that job. A minimal template batch script for running GIZMO is:

.. code:: bash

   #!/bin/bash                                                                     
   #SBATCH -J name_of_job -p normal -N 1 --ntasks-per-node 56 -t 48:00:00 -A AST21002

   source $HOME/.bashrc
   ibrun ./GIZMO ./params.txt 0 1>gizmo.out 2>gizmo.err &
   wait

But note that you can run any command in place of the ibrun command used to run gizmo above.

Key for the different options in the header:

- ``-J`` the name of the job - can be anything, but should be somewhat descriptive so you know which job is which
- ``-p`` the queue the job is being submitted to - ‘normal’ for regular jobs, but you can use ‘development’ to get a queue that will run much sooner for testing purposes, but can only run <2hr jobs one at a time.
- ``-N`` The number of nodes you want to run the job on. Each node on Frontera has 56 cores and 192GB of RAM.
- ``--ntasks-per-node`` The number of MPI processes per node - normally equal to the number of cores per node, but if you are compiling with OPENMP=<N> then divide this number by N, and add
- ``-t`` the maximum time you want the job to run for, formatted hours:minutes:seconds. The maximum for the normal queue is 48 hours, and if your run finishes before the time is up, the job will terminate.
- ``-A`` the allocation you are charging for your CPU time - here the example AST21002 is our current Frontera allocation

When you submit the job, it will wait in the queue for some amount of time until it eventually runs. To check on the status of all jobs you have submitted, use the command showq -u <your username>. There are many other arguments you can give to showq to customize the information you get. To get rich information on a certain job, use the command scontrol show jobid <job id #>
