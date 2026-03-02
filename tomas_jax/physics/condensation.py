"""TFL condensation algorithm for TOMAS-JAX (Fortran-faithful sequential version).

Phase 1: Direct port of tmcond.f and dmdt_int.f using sequential Python loops.
NOT JIT-compatible (uses Python for-loops and in-place mutation).

The TFL (Tzivion-Feingold-Levin) algorithm is a semi-Lagrangian scheme for
condensational growth/evaporation on a fixed mass grid:

    1. Construct a uniform (top-hat) distribution within each size bin
    2. Translate top-hat edges analytically using dmdt_int
    3. Remap the translated top-hats back to the fixed bin grid

This is a DISCRETE state-to-state mapping (Nk, Mk) -> (Nk_new, Mk_new),
NOT an ODE. It should be called once per condensation sub-step.

The algorithm handles multicomponent aerosols: only a single species
(``cspecies``) condenses during each call. The condensing species mass
is tracked via ``YUC``/``YLC`` (mass at top-hat edges) and remapped using
trapezoidal interpolation, while non-condensing species are remapped
proportionally.

Based on:
    - Tzivion, Feingold, Levin, JAS 1989 (TFL scheme)
    - Stevens, Feingold, Cotton, JAS 1996 (analytic growth solution)
    - Peter Adams (multicomponent extensions, AMKDRY/WR variables)
    - Win Trivitayanurak (c1/c2 correction factors, 5/25/06)

References:
    - tmcond.f lines 52-366
    - dmdt_int.f lines 27-46
"""
import numpy as np
from typing import Tuple

from ..core.config import NBINS, ICOMP, ICOMP_NODIAG, IDIAG, SRTSO4


# Constants matching Fortran PARAMETER statements
TEPS = 1.0e-40   # Threshold for TAU (growth forcing)
NEPS = 1.0e-20   # Threshold for number concentration


def dmdt_int(M0: float, TAU: float, WR: float) -> float:
    """Analytic solution to particle growth equation in mass space.

    Exact port of dmdt_int.f (Stevens et al. 1996, Eq. A3).

    Args:
        M0: Initial dry mass [kg]
        TAU: Growth forcing = (2/3)*CPT*ETA_BAR*DELTA_T
        WR: Wet/dry mass ratio

    Returns:
        Final dry mass after growth [kg]
    """
    C = 2.0 / 3.0

    MH2O = (WR - 1.0) * M0
    X = (M0 + MH2O) ** C  # L0=0
    X = max(0.0, np.sqrt(max(0.0, C * TAU + X)))
    result = X * X * X / WR  # keeps WR constant (win, 5/14/06)

    # Numerical checks (lines 42-43)
    if TAU > 0.0 and result < M0:
        result = M0
    if TAU < 0.0 and result > M0:
        result = M0

    return result


def tmcond(
    TAU: np.ndarray,
    xk: np.ndarray,
    Mkd: np.ndarray,
    Nkd: np.ndarray,
    cspecies: int,
    moxd: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Condensation step using TFL moving-center algorithm.

    Fortran-faithful sequential port of tmcond.f (lines 52-366).
    Uses Python for-loops matching the Fortran structure exactly.

    This is a DISCRETE state-to-state mapping, NOT an ODE.

    Args:
        TAU: Growth forcing per bin, shape (ibins,)
        xk: Bin boundaries [kg], shape (ibins+1,)
        Mkd: Mass concentration [kg/grid cell], shape (ibins, icomp) — INPUT
        Nkd: Number concentration [#/grid cell], shape (ibins,) — INPUT
        cspecies: Index of condensing species (0-based)
        moxd: Mass flux per particle for c1/c2 corrections, shape (ibins,)
              If None, zeros (no correction, for SO4 condensation)

    Returns:
        Nk_out: Updated number concentration, shape (ibins,)
        Mk_out: Updated mass concentration, shape (ibins, icomp)
    """
    ibins = NBINS
    icomp = ICOMP
    idiag = IDIAG

    if moxd is None:
        moxd = np.zeros(ibins)

    # Work on copies (Fortran modifies ANKD/AMKD in-place)
    ANKD = np.array(Nkd, dtype=np.float64)
    AMKD = np.array(Mkd, dtype=np.float64)
    X = np.array(xk, dtype=np.float64)  # ibins+1

    # --- Lines 82-92: Fix empty bins ---
    for k in range(ibins):
        if ANKD[k] < NEPS:
            ANKD[k] = NEPS
            AMKD[k, SRTSO4] = NEPS * 1.4 * X[k]
            for j in range(icomp):
                if j != SRTSO4:
                    AMKD[k, j] = 0.0

    # --- Lines 97-110: Transfer overflow from bins where avg > upper boundary ---
    for k in range(ibins - 1):
        AMKD_tot = 0.0
        for kk in range(icomp - idiag):
            AMKD_tot += AMKD[k, kk]
        if AMKD_tot / ANKD[k] > X[k + 1]:
            for j in range(icomp):
                AMKD[k + 1, j] += 0.1 * AMKD[k, j]
                AMKD[k, j] *= 0.9
            ANKD[k + 1] += 0.1 * ANKD[k]
            ANKD[k] *= 0.9

    # --- Lines 119-129: Initialize AMKDRY and WR ---
    AMKDRY = np.zeros(ibins)
    AMKWET = np.zeros(ibins)
    WR = np.ones(ibins)
    for L in range(ibins):
        for j in range(icomp - idiag):
            AMKDRY[L] += AMKD[L, j]
        for j in range(icomp):
            AMKWET[L] += AMKD[L, j]
        if AMKDRY[L] > 0.0:
            WR[L] = AMKWET[L] / AMKDRY[L]

    # --- Lines 139-150: Check for significant forcing ---
    maxtau = 0.0
    for L in range(ibins):
        maxtau = max(maxtau, abs(TAU[L]))

    # Output arrays
    ANK = np.zeros(ibins)
    AMK = np.zeros((ibins, icomp))

    if abs(maxtau) < TEPS:
        # No significant forcing - copy input to output
        AMK[:, :] = AMKD[:, :]
        ANK[:] = ANKD[:]
        return ANK, AMK

    # --- Main loop over source bins (lines 162-362) ---
    WW = 0.5

    for L in range(ibins):
        if ANKD[L] == 0.0:
            continue

        # If tau is zero, leave everything in same bin (lines 166-172)
        if TAU[L] == 0.0:
            ANK[L] += ANKD[L]
            for j in range(icomp):
                AMK[L, j] += AMKD[L, j]
            continue

        # --- Construct top-hat (lines 179-200) ---
        AVG = AMKDRY[L] / ANKD[L]
        XX = X[L] / AVG
        XI = 0.5 + XX * (1.5 - XX)
        if XI < 1.0:
            # Shouldn't happen; fallback: keep particles in same bin
            ANK[L] += ANKD[L]
            for j in range(icomp):
                AMK[L, j] += AMKD[L, j]
            continue

        W1 = np.sqrt(12.0 * (XI - 1.0)) * AVG
        W2 = min(X[L + 1] - AVG, AVG - X[L])
        WTH = W1 * WW + W2 * (1.0 - WW)
        XU = AVG + WTH * 0.5
        XL = AVG - WTH * 0.5

        # Ventilation (disabled: VSW=0, VNTF=0)
        TAU_L = TAU[L]

        # --- Translate top-hat edges with c1/c2 corrections (lines 219-228) ---
        YU = dmdt_int(XU, TAU_L, WR[L])
        YL = dmdt_int(XL, TAU_L, WR[L])

        if moxd[L] == 0.0:
            c1 = 1.0
        else:
            denom = YU + YL - XU - XL
            if abs(denom) < 1e-30:
                c1 = 1.0
            else:
                c1 = moxd[L] * 2.0 / denom

        sum_after = YU + YL
        if abs(sum_after) < 1e-30:
            c2 = 1.0
        else:
            c2 = c1 - (c1 - 1.0) * (XU + XL) / sum_after
        YU = YU * c2
        YL = YL * c2

        # --- Condensing species mass at edges (lines 231, 236) ---
        species_frac = AMKD[L, cspecies] / AMKDRY[L] if AMKDRY[L] > 0 else 0.0
        YUC = XU * species_frac + YU - XU
        if YU > X[ibins]:
            YUC = YUC * X[ibins] / YU
            YU = X[ibins]
        YLC = XL * species_frac + YL - XL

        if abs(YU - YL) > 0.0:
            DYI = 1.0 / (YU - YL)
        else:
            DYI = 0.0

        # --- Handle below-grid (lines 243-274) ---
        if YL < X[0]:
            ANK[L] += ANKD[L]
            for j in range(icomp):
                if j == cspecies:
                    AMK[L, j] += (YUC + YLC) * 0.5 * ANKD[L]
                else:
                    AMK[L, j] += AMKD[L, j]
            continue

        if YU < X[0]:
            continue

        # --- Remapping (lines 279-360) ---
        IMN = 0
        if TAU[L] > 0.0:
            IMN = L

        remapped = False
        for I in range(IMN, ibins):
            if YL < X[I + 1]:
                if YU <= X[I + 1]:
                    # Entire top-hat fits in bin I
                    DN = ANKD[L]
                    for j in range(icomp):
                        DM = AMKD[L, j]
                        if j == cspecies:
                            val = (YUC + YLC) * 0.5 * DN
                            if val < 0.0:
                                pass  # AMK[I,j] unchanged
                            else:
                                AMK[I, j] += val
                        else:
                            AMK[I, j] += DM
                    ANK[I] += DN
                    remapped = True
                    break
                else:
                    # Top-hat spans bin I and higher bins
                    # First partial bin (I)
                    DN = ANKD[L] * (X[I + 1] - YL) * DYI
                    for j in range(icomp):
                        DM = AMKD[L, j] * (X[I + 1] - YL) * DYI
                        if j == cspecies:
                            XP = dmdt_int(X[I + 1], -1.0 * TAU_L, WR[L])
                            YM = XP * AMKD[L, j] / AMKDRY[L] + X[I + 1] - XP if AMKDRY[L] > 0 else 0.0
                            val = (YM + YLC) * 0.5 * DN
                            if val < 0.0:
                                pass
                            else:
                                AMK[I, j] += val
                        else:
                            AMK[I, j] += DM
                    ANK[I] += DN

                    # Middle bins (K = I+1 to ibins-1)
                    found_K = None
                    for K in range(I + 1, ibins):
                        if YU <= X[K + 1]:
                            found_K = K
                            break
                        # Full bin K
                        DN = ANKD[L] * (X[K + 1] - X[K]) * DYI
                        for j in range(icomp):
                            DM = AMKD[L, j] * (X[K + 1] - X[K]) * DYI
                            if j == cspecies:
                                XP = dmdt_int(X[K], -1.0 * TAU_L, WR[L])
                                YM = XP * AMKD[L, j] / AMKDRY[L] + X[K] - XP if AMKDRY[L] > 0 else 0.0
                                AMK[K, j] += DN * 1.5 * YM
                            else:
                                AMK[K, j] += DM
                        ANK[K] += DN

                    if found_K is None:
                        # Overflow — shouldn't happen with proper grid
                        found_K = ibins - 1

                    K = found_K
                    # Last partial bin (K) — GOTO 100 target
                    DN = ANKD[L] * (YU - X[K]) * DYI
                    for j in range(icomp):
                        DM = AMKD[L, j] * (YU - X[K]) * DYI
                        if j == cspecies:
                            XP = dmdt_int(X[K], -1.0 * TAU_L, WR[L])
                            YM = XP * AMKD[L, j] / AMKDRY[L] + X[K] - XP if AMKDRY[L] > 0 else 0.0
                            AMK[K, j] += DN * (YUC + YM) * 0.5
                        else:
                            AMK[K, j] += DM
                    ANK[K] += DN
                    remapped = True
                    break
        # end I loop

    return ANK, AMK
