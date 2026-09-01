Running GIZMO on other machines
-------------------------------

To run on a machine without an entry in these docs:

- Check ``Makefile.systype`` for an existing ``SYSTYPE`` for your machine and un-comment it. If there is none, add a new ``SYSTYPE`` block to the Makefile specifying the compilers, flags, and library paths for your system — it is usually easiest to copy an existing entry for a similar machine. See the `GIZMO documentation <http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html>`__ for details.
- GIZMO requires an MPI implementation, HDF5, GSL, and FFTW; on most clusters these are provided as modules that must be loaded at compile- and run-time (typically in your ``.bashrc`` or batch script).
- Optimal ``OPENMP`` settings and MPI process placement/affinity are machine-dependent — benchmark a short job at a few settings before committing a large run.
