GIZMO/STARFORGE Version History
================================

This page summarizes physics-relevant changes between tagged GIZMO versions for
STARFORGE simulations run with ``SINGLE_STAR_STARFORGE_DEFAULTS`` plus full feedback
and cooling.  Changes are grouped by topic; commit hashes are cited for traceability.

The three canonical tags are:

* **v1.1** — commit ``8d56d9c3`` (2021-12-16)
* **v1.2** — commit ``30459741`` (2023-07-22)
* **v1.3** — commit ``168cdb3d``

----

Pre-v1.1
---------

Ionizing-photon transport bugfix (``3ae730f1``, 2021-12-14)
   A one-line fix to the M1 face-flux gate in ``radiation/rt_diffusion_explicit.h``::

       - if((scalar_i>0)&&(scalar_j>0)&&...)
       + if((scalar_i+scalar_j>0)&&...)

   The old code required **both** cells across a face to have non-zero photon energy,
   so the I-front could never propagate into previously-empty neutral gas.  HII-region
   expansion in STARFORGE runs predating 2021-12-14 was effectively frozen.
   A companion commit (``4dc827f5``) floors ``Rad_E_gamma`` at ``MIN_REAL_NUMBER`` to
   prevent any code path from re-encountering the same pathology.

Self-consistent RT boundary condition (``a40f791e``, 2021-06-17)
   Introduced ``rt_apply_boundary_conditions``: cells within 10 % of the box edge have
   ``Rad_E_gamma[k]`` clamped to band-resolved ISRF values each step.  Already present
   in v1.1; earlier runs saw M1 photons hitting the box edge as vacuum.

----

v1.1
----

``GALSF_USE_SNE_ONELOOP_SCHEME`` forced on (``8d56d9c3``)
   The v1.1 head commit itself forces the FIRE-2 one-loop mechanical-feedback coupling
   for any run with ``SINGLE_STAR_FB_WINDS`` or ``SINGLE_STAR_FB_SNE``.  Unchanged
   through v1.3 — all three versions use 1-loop coupling.

``EOS_SUBSTELLAR_ISM`` on but **buggy**
   The flag was a default in v1.1 but implemented the *integrated* γ using the
   *differential* γ formula from Vaidya 2015, producing unphysical behaviour in the
   molecular regime.  Disabled in v1.2; correctly reimplemented in v1.3.

Reduced speed of light default: ``1.0e-4``

----

v1.1 → v1.2
------------

Radiation / RT
~~~~~~~~~~~~~~

Dust temperature and IR opacity refactor
   Emission opacity (depends only on T\ :sub:`dust`) separated from absorption opacity
   (depends on T\ :sub:`rad` and T\ :sub:`dust`).  New routines ``rt_eqm_dust_temp()``
   and ``gas_dust_heating_coeff()``.  Improved low-T (<10 K) opacity extrapolation with
   a β=2 law (relevant for opacity-limit collapse).

Reduced speed of light raised from ``1e-4`` to ``3e-4`` (``d2b68c2f``)
   3× increase.  Affects I-front propagation speed, M1 timestep cost, and
   radiation-pressure-driven dynamics.  Effective only if not overridden in Config.sh.

RT photon re-injection after accretion (``2b9dbc9e``, then made default in ``56944088``)
   When a sink swallows a cell, the accreted ``Rad_E_gamma`` is stored and re-emitted
   on the next RT source step in the IR bin.  Corrects a systematic photon loss in
   optically-thick accretion envelopes at low RSOL.  Made a default for
   ``SINGLE_STAR_STARFORGE_DEFAULTS`` with ``RT_INFRARED`` in ``56944088``.  Not
   present at all in v1.1.

Ströomgren-radius criterion bugfix (``24109dbc``)
   ``rt_sourceinjection_evaluate`` was comparing neighbor density instead of the star's
   local density to decide whether to downgrade photons to IR.  Shells surrounding HII
   regions were never ionized under the old code.

Cosmic-ray background and attenuation
   Three staged commits build the current prescription:

   1. ``1396c4bc`` — CR energy density scaled linearly with ``InterstellarRadiationFieldStrength``
      (note: a typo ``RT_ISRF_BRACKGROUND`` silently broke the H₂ branch).
   2. ``877d35dd`` — Switched to √ISRF scaling (ζ\ :sub:`CR` ∝ Σ\ :sub:`SFR`\ :sup:`0.5`,
      Crocker+20/Krumholz+23); CR coupling consolidated into a single function.
   3. ``87599421`` — Column-density attenuation added: flat for N\ :sub:`H` < 10²¹ cm⁻²,
      then N\ :sub:`H`\ :sup:`-1` power law with exponential cutoff above ~100 g cm⁻².

   Net effect: dense cloud cores get lower ζ\ :sub:`CR` → higher molecular fraction and
   less CR heating compared with v1.1.

Sink particles and feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Protostellar evolution rewritten (``9cd3738f`` et al.)
   Now uses the Nakano-2000 evolution equation with accretion-rate-dependent
   interpolation and Hermite-style subcycling.  Sets R\ :sub:`*`, T\ :sub:`*`,
   L\ :sub:`acc`, L\ :sub:`MS` — feeds directly into RT luminosity, wind Ṁ, and
   photoionization rates.  Quantitatively different T/L tracks vs. v1.1.

``SINGLE_STAR_DIRECT_GRAVITY_RADIUS = 1000 AU`` as default (``56944088``)
   All star–star gravity within 1000 AU is computed exactly.  Changes binary/multiple
   dynamics; expect tighter binaries and different multiplicity statistics.

Sink-formation criteria at high density (``d1ce905e`` + ``93ed5116``)
   * Prevents spurious twin-sink formation within 0.1 AU of an existing sink.
   * Fixes an overflow in the optically-thick column-density expression above
     10¹⁴ cm⁻³ that pushed cooling into wrong logic.
   * Fixed flag logic so the optically-thick virial limiter (``v_fast → 0.2 km/s``
     above n > 10¹³ cm⁻³) activates when ``RT_INFRARED`` is on.  In v1.1 this branch
     was silently disabled for RT runs, making sink formation very slow in dense cores.

Sink-accretion timestep indexing fix (``8bebf956``)
   In ``BH_INTERACT_ON_GAS_TIMESTEP`` mode, four routines used the BlackholeTempInfo
   index instead of the particle index to fetch ``dt_since_last_gas_search``.
   Mdot histories before this fix can be off by a factor of a few.

Spawning logic bug (``4892f8ad``)
   ``Max_Unspawned_MassUnits_fromSink`` was compared in cell-count units against a
   mass in mass units, giving a wrong trigger threshold for jet/wind spawning.

Equal-mass sink merging fix (``51c94489``)
   Strict inequality prevented equal-mass sinks from merging; edge case for symmetric
   ICs or twin binaries.

Massive-star timestep before SN (``517c40fa``)
   Ensures massive stars do not skip gas-interaction steps immediately before going SN,
   preventing unphysical single-step dump of accumulated ejecta mass.

``BH_Mass`` bookkeeping in ``SINGLE_STAR_FB_WINDS`` (``d8d122ad``)
   In the local area-weighted coupling path, ``BH_Mass`` was not decremented alongside
   ``Mass`` when wind mass was lost, causing luminosity/wind-rate calculations to drift.

Spawned-wind direction seeding bug (``e573db86``)
   The angular-grid spawn mode was keying off ``ID_child_number`` vs. ``ID_generation``
   with a swapped convention from an earlier commit, misaligning the basis-rotation
   cadence for jet spawning.

``EOS_SUBSTELLAR_ISM`` **disabled** (``b53e7e1c``, 2022-10-27)
   Deliberately removed from STARFORGE defaults because the v1.1 implementation used
   the wrong γ formula.  v1.2 falls back to ideal-gas γ = 5/3 in the molecular regime.

----

v1.2 → v1.3
------------

Thermodynamics
~~~~~~~~~~~~~~

``EOS_SUBSTELLAR_ISM`` correctly reinstated
   Now uses the self-consistent molecular+atomic EOS with full T-dependent γ(T,ρ).
   H₂ physics moved to ``hydrogen_molecule.c``.  Expect lower gas temperatures at
   densities where H₂ matters compared with v1.2's ideal-gas fallback.

``EOS_PRECOMPUTE`` infrastructure
   New ``set_eos`` routine caches T, γ, P; ``get_pressure`` is now a getter.
   ``convert_u_to_temp`` rewritten with a Newton iteration using the analytic heat
   capacity, plus secant fallback.

Bracketed root-finder (``bracketed_rootfind.h``)
   Applied to the fH₂ solver, the dust-temperature solver, and the top-level cooling
   iteration.  Faster convergence and better diagnostics on failure.

Dust solver improvements
   Gas–dust–radiation coupling improved for the optically-thick core regime.  Complete
   dust sublimation now assigns ``MAX_DUST_TEMP`` (10⁴ K) and zeroes dust
   abundances/processes.  CO cooling now uses a Whitworth-2018 LVG optically-thick
   limiter.

Stellar feedback
~~~~~~~~~~~~~~~~

Wind mass-loss rates use main-sequence luminosity only (``b07a4b5c`` + related)
   Wind Ṁ is now driven by the *main-sequence* luminosity, not total (MS + accretion)
   luminosity.  The model fits do not include accretion luminosity; actively accreting
   massive protostars have weaker winds in v1.3 than in v1.2.

Cosmic rays
~~~~~~~~~~~

Radioactive-decay floor on ζ\ :sub:`CR` (``8e36c561``)
   Adds ``+= 1e-21 · (Z/Z☉)`` to ``Get_CosmicRayIonizationRate_cgs``, providing a
   metallicity-scaled minimum ζ\ :sub:`CR` in deeply shielded gas where column
   attenuation otherwise drives it to zero.  Applies to all ``METALS`` runs (broader
   than the commit message implies).

CR energy density returned 0 for particle index 0 (``7fee2822``)
   Off-by-one: ``if(i<=0)`` → ``if(i<0)``.  The first particle on each MPI rank always
   got u\ :sub:`CR` = 0 in v1.2, causing stochastic single-cell anomalies in H₂
   chemistry and CR heating after every domain decomposition.

Other
~~~~~

``z_background`` / ``Redshift_RT_Background`` parameterfile entry
   Runtime control of the redshift used to compute the background UV/IR in
   ``RT_ISRF_BACKGROUND``.

----

Summary: what changed between each version pair
------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Transition
     - Biggest physics changes for a typical STARFORGE run
   * - pre-v1.1 → v1.1
     - Ionizing-photon I-front transport fixed; 1-loop wind coupling forced on;
       RT boundary condition in place.
   * - v1.1 → v1.2
       rewrite (new L*/R* tracks); sink-formation criteria unlocked for IR-RT runs;
       RSOL raised 3×; photon re-injection after accretion added; EOS_SUBSTELLAR_ISM
       disabled; CR column attenuation introduced.
   * - v1.2 → v1.3
     - EOS_SUBSTELLAR_ISM correctly implemented (molecular-regime thermodynamics);
       root-finder rewrite (dust, fH₂, cooling solvers); wind input luminosity changed
       to MS-only; CR floor added; per-rank CR=0 bug fixed.