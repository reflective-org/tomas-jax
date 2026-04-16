"""Physical parameterizations for aerosol microphysics.

Coagulation:
    coagulation_kernel  - Brownian coagulation kernel (Fuchs correction)
    coagulation_rates   - TFL mass-conserving coagulation rates
    properties          - Particle diameter, diffusivity, thermal speed
    density             - Mixed-salt aerosol density (Tang 1997)

Condensation:
    gas_properties      - Gas diffusivity, MFP, Fuchs-Sutugin correction
    condensation_sink   - First-order gas loss rate (CS) to aerosol
    condensation        - TFL condensation algorithm (dmdt_int + tmcond)
    condensation_ppm    - PPM condensation algorithm (Eulerian advection, JIT-compatible)
    ezcond              - Simple condensation driver with sub-stepping (TFL)
    ezcond_ppm          - PPM-aware condensation driver (numpy wrapper)
    ezcond_ppm_jax      - Pure-JAX PPM condensation driver (JIT-compilable)

Equilibrium:
    nh3_equilibrium     - NH3/NH4 stoichiometric partitioning
    water_equilibrium   - Hygroscopic water uptake (ISORROPIA fits)

Radiative:
    bhmie               - Bohren-Huffman Mie scattering (Bohren & Huffman 1983)
    radiative_forcing   - Direct SW radiative forcing (Chylek & Wong 1995, Pierce et al. 2010)
"""
