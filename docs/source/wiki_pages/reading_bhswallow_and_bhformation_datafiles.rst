ASCII output file formats
~~~~~~~~~~~~~~~~~~~~~~~~~

The bhswallow and bhformation data files are ascii data files written by GIZMO as the simulation runs, each time there is an accretion or sink formation event.

bhswallow (OLD FORMAT BEFORE 3/20/2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   # (0) Time
   # (1) ID
   # (2) Accretor Sink Mass
   # (3) X
   # (4) Y
   # (5) Z
   # (6) Accreta ID
   # (7) Accreta mass
   # (8) dx
   # (9) dy
   # (10) dz
   # (11) dvx
   # (12) dvy
   # (13) dvz
   # (14) Accreta specific internal energy
   # (15) Bx
   # (16) By
   # (17) Bz
   # (18) Density

bhswallow (NEW FORMAT AFTER 3/20/2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   # (0) Time
   # (1) Accretor ID
   # (2) Accretor Child ID
   # (3) Accretor Generation ID
   # (4) Accretor Sink Mass
   # (5) X
   # (6) Y
   # (7) Z
   # (8) Accreta ID
   # (9) Accreta Child ID
   # (10) Accreta Generation ID
   # (11) Accreta mass
   # (12) dx
   # (13) dy
   # (14) dz
   # (15) dvx
   # (16) dvy
   # (17) dvz
   # (18) Accreta specific internal energy
   # (19) Bx
   # (20) By
   # (21) Bz
   # (22) Density

bhformation
^^^^^^^^^^^

::

   (0) Time
   (1) ID
   (2) Mass
   (3) X
   (4) Y
   (5) Z
   (6) vx
   (7) vy
   (8) vz
   (9) Bx
   (10) By
   (11) Bz
   (12) u
   (13) rho
   (14) cs
   (15) cell size
   (16) column density
   (17) velocity gradient^2
   (18) min dist to nearest star


SN_details_STARFORGE.txt records each supernova event.

SN_details_STARFORGE
^^^^^^^^^^^^^^^^^^^^

::
  (0) Time
  (1) ID
  (2) Mass
  (3) X
  (4) Y
  (5) Z    
  (6) VX
  (7) VY
  (8) VZ
