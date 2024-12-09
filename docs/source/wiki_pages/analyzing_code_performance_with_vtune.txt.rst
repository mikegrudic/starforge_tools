Analyzing code performance with VTune
-------------------------------------

Intel’s VTune utility is a powerful tool for determining where a code is spending most of its time, and hence which parts are best to focus on for optimization. Here I document the workflow for analyzing a GIZMO STARFORGE simulation on Frontera.

Compiling
^^^^^^^^^

When compiling with the Intel compiler, you must compile with the ``-g`` flag to enable the logging capabilities needed by VTune. You must modify your Makefile to look something like this:

|screenshot_from_2023-06-05_12-01-46.png|

Running the code with the VTune command line interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VTune has a large number of different options for what data the analysis gathers, and it is possible to configure arbitrary analyses. For our purposes there are three main default analyses that are useful: ``hotspots``, ``threading``, and ``hpc-performance``. Let‘s assume we’re running a ’‘hotspots’’ analysis. Then your command to run the code with vtune active will look like

<WRAP prewrap>

.. code:: bash

   module load vtune
   vtune -collect hotspots ibrun ./GIZMO params.txt 0>gizmo.out 1>gizmo.err

</WRAP>

The code will then run and output data to a directory that looks like ``rXXXhs`` where XXX is some number.

At this point you can generate a text report of the performance metrics in the command line as described `here <https://www.nas.nasa.gov/hecc/support/kb/finding-hotspots-in-your-code-with-the-intel-vtune-command-line-interface_506.html>`__, but VTune is most powerful when you can use its GUI. The GUI is slow over X so this is best done on your local machine. rsync the log directory to your machine, then open the data in VTune with ctrl+O and opening the file ``rXXXhs.vtune`` in the directory. You will then be able to view the performance summary, and analyze the performance throughout the call stack in different ways in the various tabs: |screenshot_from_2023-06-05_12-51-20.png| |screenshot_from_2023-06-05_12-55-46.png|

.. |screenshot_from_2023-06-05_12-01-46.png| image:: /screenshot_from_2023-06-05_12-01-46.png
   :width: 400px
.. |screenshot_from_2023-06-05_12-51-20.png| image:: //screenshot_from_2023-06-05_12-51-20.png
   :width: 600px
.. |screenshot_from_2023-06-05_12-55-46.png| image:: //screenshot_from_2023-06-05_12-55-46.png
   :width: 600px