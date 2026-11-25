Setting up TURBSPHERE runs
--------------------------

A TURBSPHERE setup is run in two stages: an initialization run that stirs the cloud to statistical equilibrium, followed by the star formation (SF) run.

The stirring run GIZMO binary is compiled with flags:

::

           SINGLE_STAR_STARFORGE_DEFAULTS
           MAGNETIC
           COOLING
           SINGLE_STAR_FB_JETS
           SINGLE_STAR_FB_RAD
           SINGLE_STAR_FB_WINDS
           SINGLE_STAR_FB_SNE
           BOX_PERIODIC
           GRAVITY_NOT_PERIODIC
           ADAPTIVE_TREEFORCE_UPDATE=0.0625
           OPENMP=2
           STARFORGE_GMC_TURBINIT

and the value of RT_SPEEDOFLIGHT_REDUCTION was reduced by a factor of 3 in allvars.h (purely for speed, this is optional and should not be used for the eventual SF run). To perform the stirring run, start with the desired SPHERE IC and scale the turbulence driving parameters in params.txt appropriately (e.g. rescale all wavelengths to the cloud size, tune vRMS to get the desired Mach number, scale the coherence time to the cloud crossing time). The equilibrium turbulence properties will be converged at very modest resolution, so it’s easiest to tune at low resolution before running the real thing. **It is highly recommended to pre-tune the turbulence parameters at low resolution before running your full-resolution stirring run.**

Then to run the SF run, simply disable STARFORGE_GMC_TURBINIT, and optionally add TURB_DRIVING if you want driving to continue. The SF run should ideally be treated as a flag 2 snapshot-restart of the stirring run, as otherwise the radiation field will be re-initialized to the ISRF instead of the statistical equilibrium attained during the stirring.
