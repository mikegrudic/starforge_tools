Machine-specific notes, tips, and submit scripts
------------------------------------------------

Aitken
~~~~~~

To use Aitken, follow instructions for logging into Pleiades on the NAS website - the machines share login nodes.

Aitken: Cascade Lake nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Compiling GIZMO
'''''''''''''''

To compile GIZMO for Aitken Cascade Lake nodes, uncomment the ``Pleiades`` line in ``Makefile.systype`` and load the following (can add this to your ``.profile`` to run automatically upon login):

.. code:: bash

   module load comp-intel mpi-hpe/mpt pkgsrc/2021Q2                                                                                            
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PKGSRC_BASE/lib

| 

Optimal Parallelization
'''''''''''''''''''''''

Aitken has 2 types of nodes: Intel Cascade Lake and AMD Rome. Currently only Cascade Lake nodes have been tested, which have 2 CPUs with 20 physical cores each, with 2 hyperthreads each for a total of 80 possible concurrent processes per node. **However** quite often GIZMO is limited by cache/memory throughput rather than instruction throughput, so running the maximum number of processes is not always optimal.

Testing a variety of MPI and hybrid MPI/OpenMP configurations on the slowest (most load-imbalanced) part of a typical mid-sized (2e7 gas cell) run, the optimal node configuration appeared to be 20 MPI ranks per node with 2 OpenMP threads per rank (i.e. compile with ``OPENMP=2``).

Submit script
'''''''''''''

Example submit script for a STARFORGE run on 8 Aitken Cascade Lake node with 20 MPI ranks per node and 2 OpenMP threads per MPI rank:

.. code:: bash

   #PBS -l select=8:ncpus=40:mpiprocs=20:model=cas_ait
   #PBS -l walltime=120:00:00
   #PBS -q long

   source ~/.profile
   export OMP_NUM_THREADS=2

   export MPI_DSM_DISTRIBUTE=0
   export KMP_AFFINITY=disabled

   mpiexec -np 160 omplace -nt 2 ./GIZMO params.txt 2 1>gizmo.out 2> gizmo.err
   wait

| 