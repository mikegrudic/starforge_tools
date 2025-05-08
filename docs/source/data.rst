
**********************************
Data Fields In STARFORGE Snapshots
**********************************

The basic dataset of a STARFORGE simulation consists of a set of chronologically-ordered snapshots 

Gas Data 
========
In GIZMO snapshots, particle type 0 always represents the gas elements in the simulation.

What is a gas cell in GIZMO MFM/MFV simulation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is actually kind of a tricky question. In a finite-volume grid code, the cells represent real, geometrically-demarcated sub-volumes of the simulation domain. In an SPH simulation, the gas particles represent physical blobs of gas interacting through a force law. GIZMO MFM/MFV gas cells are neither. The data structure used is similar to the particle list in SPH, but the actual way conservation laws are solved is much more closely related to the finite-volume code.

An easier concept to wrap one's head around is the Voronoi tesselation: given a set of mesh-generating points, the Voronoi tesselation is the set of sub-volumes of the domain consisting of points grouped by their closest mesh-generating point. In a Voronoi tesselation, 100% of the "weight" of each point in space is assigned to the nearest mesh-generating point (see panel 2 below).

.. image:: media/method_demo.jpeg

The discretization used in GIZMO's MFM/MFV methods is a generalization of the Voronoi tesselation. Instead of associating 100% of each point in space with one mesh-generating, a *weighting function* is defined to decide what *fraction* of each point belongs to each mesh-generating point. A pretty accurate way to think of it is as a Voronoi tesselation with blurry boundaries (panel 1 above). Then, the distinction between MFM and MFV has to do with what is assumed about the motion of the points: the MFM method (used in all current STARFORGE simulations) is constructed so that the points move in a way where the different subdomains exchange 0 mass in a timestep. This leads to a quasi-Lagrangian method that allows us to follow gas elements of fixed mass.

This details matter for how the simulation is run, but once we have the outputs it's usually fine to think of the gas cells as a collection of particles and analyze and interpret the data as you would any SPH particle dataset.

Documentation of data fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``PartType0/Coordinates``
``/PartType0/Coordinates``
``/PartType0/Density``
``/PartType0/DustToGasRatio_Local``
``/PartType0/Dust_Temperature``
``/PartType0/ElectronAbundance``
``/PartType0/HII``
``/PartType0/IRBand_Radiation_Temperature Dataset``
``/PartType0/InternalEnergy``
``/PartType0/MagneticField``
``/PartType0/Masses``
``/PartType0/Metallicity``
``/PartType0/MolecularMassFraction``
``/PartType0/NeutralHydrogenAbundance``
``/PartType0/ParticleChildIDsNumber``
``/PartType0/ParticleIDGenerationNumber``
``/PartType0/ParticleIDs``
``/PartType0/PhotonEnergy``
``/PartType0/PhotonFluxDensity``
``/PartType0/Potential``
``/PartType0/Pressure``
``/PartType0/SmoothingLength``
``/PartType0/Temperature``
``/PartType0/Velocities``


=========
/PartType5               Group
/PartType5/BH_AccretionLength Dataset {117}
/PartType5/BH_Mass       Dataset {117}
/PartType5/BH_Mass_AlphaDisk Dataset {117}
/PartType5/BH_Mdot       Dataset {117}
/PartType5/BH_NProgs     Dataset {117}
/PartType5/BH_Specific_AngMom Dataset {117, 3}
/PartType5/Coordinates   Dataset {117, 3}
/PartType5/DustToGasRatio_Local Dataset {117}
/PartType5/Mass_D        Dataset {117}
/PartType5/Masses        Dataset {117}
/PartType5/Metallicity   Dataset {117, 14}
/PartType5/ParticleChildIDsNumber Dataset {117}
/PartType5/ParticleIDGenerationNumber Dataset {117}
/PartType5/ParticleIDs   Dataset {117}
/PartType5/Potential     Dataset {117}
/PartType5/ProtoStellarAge Dataset {117}
/PartType5/ProtoStellarRadius_inSolar Dataset {117}
/PartType5/ProtoStellarStage Dataset {117}
/PartType5/SinkInitialMass Dataset {117}
/PartType5/SinkRadius    Dataset {117}
/PartType5/StarLuminosity_Solar Dataset {117}
/PartType5/StellarFormationTime Dataset {117}
/PartType5/Velocities    Dataset {117, 3}
/PartType5/ZAMS_Mass     Dataset {117}