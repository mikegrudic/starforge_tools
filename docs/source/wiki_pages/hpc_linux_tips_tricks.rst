HPC/Linux Tips & Tricks
-----------------------

This page compiles various routine tasks useful for working with the simulations in HPC environments, that are small enough to be explained in a 1-2 paragraphs.

Transferring files without having to stay logged in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When moving around large datasets it is useful to keep the transfer going even if you want to log out. To do so, start running your usual transfer command:

.. code:: bash

   scp -r username@my.hpc.cluster.edu:source_path destination_path

or

.. code:: bash

   rsync -av -e ssh username@my.hpc.cluster.edu:source_path destination_path

| 
| While the transfer is running, stop it with ``CTRL+Z``. Then, restart it in the background with ``bg``. Finally, to keep the process running after you log off, run ``disown -h``.

Your tip/trick here
~~~~~~~~~~~~~~~~~~~
