"""Bohren-Huffman Mie scattering calculation.

Compute scattering and extinction efficiencies for homogeneous spheres
using the exact Lorenz-Mie theory series expansion.

Ported from the classic Bohren & Huffman (1983) algorithm.
Reference implementation: references/radiative-forcing/bhmie.py

Provides three variants:
    - ``bhmie``: Original numpy implementation (reference/validation).
    - ``bhmie_jax``: Full JAX port (JIT-compilable, vmappable, differentiable).
      Returns S1, S2, Qext, Qsca, Qback, gsca — identical to numpy version.
    - ``bhmie_qsca_jax``: Lightweight JAX variant that skips angular
      amplitudes (S1/S2) and returns only (Qext, Qsca, gsca). Faster for
      precomputation where scattering efficiencies are all that's needed.

Reference:
    Bohren, C. F. and D. R. Huffman (1983), Absorption and Scattering
    of Light by Small Particles, Wiley-Interscience.
"""
import numpy as np
from functools import partial
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp

# Fixed upper bounds for JIT tracing.
# TOMAS 40-bin grid: max size parameter x ≈ 164 → nstop ≈ 187.
# 80-bin grid: x_max ≈ 174 → nstop ≈ 197. NSTOP_MAX=250 gives headroom.
NSTOP_MAX = 250
NMX_MAX = 300


def bhmie(x, refrel, nang=2):
    """Bohren-Huffman Mie scattering for a homogeneous sphere.

    Args:
        x: Size parameter = 2*pi*radius / wavelength (dimensionless).
        refrel: Complex refractive index of sphere relative to medium
            (e.g. 1.5+0.02j).
        nang: Number of angles for S1 and S2 in range [0, pi/2].
            Must be >= 2.

    Returns:
        S1: Complex scattering amplitude function, shape (2*nang-1,).
            Angles from 0 to pi.
        S2: Complex scattering amplitude function, shape (2*nang-1,).
        Qext: Extinction efficiency.
        Qsca: Scattering efficiency.
        Qback: Backscattering efficiency.
        gsca: Asymmetry parameter <cos(theta)>.
    """
    nmxx = 150000

    if nang > 1000:
        raise ValueError("nang > 1000 in bhmie")
    if nang < 2:
        nang = 2

    s1_1 = np.zeros(nang, dtype=np.complex128)
    s1_2 = np.zeros(nang, dtype=np.complex128)
    s2_1 = np.zeros(nang, dtype=np.complex128)
    s2_2 = np.zeros(nang, dtype=np.complex128)
    pi = np.zeros(nang)
    tau = np.zeros(nang)

    pii = 4.0 * np.arctan(1.0)
    dx = float(x)
    y = x * refrel
    ymod = abs(y)

    # Series terminated after NSTOP terms
    xstop = x + 4.0 * x**0.3333 + 2.0
    nmx = max(xstop, ymod) + 15.0
    nmx = int(nmx)
    nstop = int(xstop)

    if nmx > nmxx:
        raise ValueError(f"nmx > nmxx={nmxx} for |m|x={ymod}")

    dang = 0.5 * pii / (nang - 1)
    amu = np.cos(np.arange(nang) * dang)

    pi0 = np.zeros(nang)
    pi1 = np.ones(nang)

    # Logarithmic derivative D(J) by downward recurrence
    nn = nmx - 1
    d = np.zeros(nn + 1, dtype=np.complex128)
    for n in range(nn):
        en = nmx - n
        d[nn - n - 1] = (en / y) - (1.0 / (d[nn - n] + en / y))

    # Riccati-Bessel functions by upward recurrence
    psi0 = np.cos(dx)
    psi1 = np.sin(dx)
    chi0 = -np.sin(dx)
    chi1 = np.cos(dx)
    xi1 = psi1 - chi1 * 1j
    qsca = 0.0
    gsca = 0.0
    p = -1

    for n in range(nstop):
        en = n + 1.0
        fn = (2.0 * en + 1.0) / (en * (en + 1.0))

        psi = (2.0 * en - 1.0) * psi1 / dx - psi0
        chi = (2.0 * en - 1.0) * chi1 / dx - chi0
        xi = psi - chi * 1j

        if n > 0:
            an1 = an
            bn1 = bn

        an = (d[n] / refrel + en / dx) * psi - psi1
        an = an / ((d[n] / refrel + en / dx) * xi - xi1)
        bn = (refrel * d[n] + en / dx) * psi - psi1
        bn = bn / ((refrel * d[n] + en / dx) * xi - xi1)

        qsca += (2.0 * en + 1.0) * (abs(an)**2 + abs(bn)**2)
        gsca += ((2.0 * en + 1.0) / (en * (en + 1.0))) * (
            an.real * bn.real + an.imag * bn.imag
        )

        if n > 0:
            gsca += ((en - 1.0) * (en + 1.0) / en) * (
                an1.real * an.real + an1.imag * an.imag
                + bn1.real * bn.real + bn1.imag * bn.imag
            )

        # Scattering intensity: angles 0 to pi/2
        pi = pi1.copy()
        tau = en * amu * pi - (en + 1.0) * pi0
        s1_1 += fn * (an * pi + bn * tau)
        s2_1 += fn * (an * tau + bn * pi)

        # Angles pi/2 to pi (symmetry)
        p = -p
        s1_2 += fn * p * (an * pi - bn * tau)
        s2_2 += fn * p * (bn * pi - an * tau)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1 - chi1 * 1j

        # pi_{n+1} from pi_n and pi_{n-1}
        pi1 = ((2.0 * en + 1.0) * amu * pi - (en + 1.0) * pi0) / en
        pi0 = pi.copy()

    # Assemble full-range S1, S2 (0 to pi)
    S1 = np.concatenate((s1_1, s1_2[-2::-1]))
    S2 = np.concatenate((s2_1, s2_2[-2::-1]))

    gsca = 2.0 * gsca / qsca
    qsca = (2.0 / (dx * dx)) * qsca
    qext = (4.0 / (dx * dx)) * S1[0].real
    qback = 4.0 * (abs(S1[2 * nang - 2]) / dx)**2

    return S1, S2, qext, qsca, qback, gsca


# =========================================================================
# JAX implementations
# =========================================================================

def _compute_nstop(x):
    """Compute series termination index (traced, not used for loop bounds)."""
    return jnp.int32(x + 4.0 * x**0.3333 + 2.0)


@partial(jax.jit, static_argnums=(2,))
def bhmie_jax(x, refrel, nang=2):
    """JAX port of Bohren-Huffman Mie scattering — JIT-compilable.

    Functionally identical to ``bhmie`` but uses ``jax.lax.fori_loop``
    for JIT tracing. All loops run to fixed upper bounds (NSTOP_MAX,
    NMX_MAX) with masking for the actual termination index.

    Args:
        x: Size parameter (scalar, traced).
        refrel: Complex refractive index (scalar, traced).
        nang: Number of angles (static, determines array shapes).

    Returns:
        Same 6-tuple as ``bhmie``: (S1, S2, Qext, Qsca, Qback, gsca).
    """
    dx = x * 1.0  # ensure float
    y = x * refrel
    nstop = _compute_nstop(x)

    dang = 0.5 * jnp.pi / (nang - 1)
    amu = jnp.cos(jnp.arange(nang) * dang)

    # --- Downward recurrence for logarithmic derivatives d[n] ---
    # Uses static indexing (k from NMX_MAX-2 down to 0) for clean autodiff.
    # Matches original by zeroing d[k] for k >= nmx (boundary d[nmx-1]=0).
    nmx = jnp.int32(jnp.maximum(x + 4.0 * x**0.3333 + 2.0,
                                  jnp.abs(y)) + 15.0)
    d = jnp.zeros(NMX_MAX, dtype=jnp.complex128)

    def recurrence_body(i, d):
        k = NMX_MAX - 2 - i
        en = jnp.float64(k + 2)
        new_val = (en / y) - (1.0 / (d[k + 1] + en / y))
        # Only fill indices below nmx-1 (boundary: d[nmx-1]=0)
        d = d.at[k].set(jnp.where(k < nmx - 1, new_val, 0.0 + 0j))
        return d

    d = jax.lax.fori_loop(0, NMX_MAX - 1, recurrence_body, d)

    # --- Main Mie series loop ---
    # carry: (psi0, psi1, chi0, chi1, xi1, qsca, gsca,
    #         an, bn, an_prev, bn_prev, p,
    #         s1_1, s1_2, s2_1, s2_2, pi0_arr, pi1_arr)
    psi0_init = jnp.cos(dx)
    psi1_init = jnp.sin(dx)
    chi0_init = -jnp.sin(dx)
    chi1_init = jnp.cos(dx)
    xi1_init = psi1_init - chi1_init * 1j

    zero_c = jnp.complex128(0.0)
    init_carry = (
        psi0_init, psi1_init, chi0_init, chi1_init, xi1_init,
        0.0, 0.0,                                  # qsca, gsca
        zero_c, zero_c, zero_c, zero_c,            # an, bn, an_prev, bn_prev
        -1.0,                                       # p (alternating sign)
        jnp.zeros(nang, dtype=jnp.complex128),      # s1_1
        jnp.zeros(nang, dtype=jnp.complex128),      # s1_2
        jnp.zeros(nang, dtype=jnp.complex128),      # s2_1
        jnp.zeros(nang, dtype=jnp.complex128),      # s2_2
        jnp.zeros(nang),                             # pi0
        jnp.ones(nang),                              # pi1
    )

    def mie_body(n, carry):
        (psi0, psi1, chi0, chi1, xi1, qsca, gsca,
         an, bn, an_prev, bn_prev, p,
         s1_1, s1_2, s2_1, s2_2, pi0_arr, pi1_arr) = carry

        en = jnp.float64(n + 1)
        fn = (2.0 * en + 1.0) / (en * (en + 1.0))
        active = n < nstop

        # Riccati-Bessel recurrence — freeze when inactive to prevent overflow
        psi = jnp.where(active, (2.0 * en - 1.0) * psi1 / dx - psi0, psi1)
        chi = jnp.where(active, (2.0 * en - 1.0) * chi1 / dx - chi0, chi1)
        xi = psi - chi * 1j

        # Save previous an, bn before computing new ones
        an_prev_new = an
        bn_prev_new = bn

        # Mie coefficients — guard denominators to prevent NaN in gradients
        d_n = d[jnp.clip(n, 0, NMX_MAX - 1)]
        numer_an = (d_n / refrel + en / dx) * psi - psi1
        denom_an = (d_n / refrel + en / dx) * xi - xi1
        numer_bn = (refrel * d_n + en / dx) * psi - psi1
        denom_bn = (refrel * d_n + en / dx) * xi - xi1
        safe_denom_an = jnp.where(active, denom_an, 1.0 + 0j)
        safe_denom_bn = jnp.where(active, denom_bn, 1.0 + 0j)
        an_new = jnp.where(active, numer_an / safe_denom_an, 0.0 + 0j)
        bn_new = jnp.where(active, numer_bn / safe_denom_bn, 0.0 + 0j)

        # Accumulate qsca
        dq = (2.0 * en + 1.0) * (jnp.abs(an_new)**2 + jnp.abs(bn_new)**2)

        # Accumulate gsca: first term (always)
        dg1 = ((2.0 * en + 1.0) / (en * (en + 1.0))) * (
            an_new.real * bn_new.real + an_new.imag * bn_new.imag)
        # Second term (n > 0 only)
        dg2 = jnp.where(
            n > 0,
            ((en - 1.0) * (en + 1.0) / en) * (
                an_prev_new.real * an_new.real + an_prev_new.imag * an_new.imag
                + bn_prev_new.real * bn_new.real + bn_prev_new.imag * bn_new.imag),
            0.0)

        qsca = jnp.where(active, qsca + dq, qsca)
        gsca = jnp.where(active, gsca + dg1 + dg2, gsca)

        # Angular functions
        pi_arr = pi1_arr
        tau_arr = en * amu * pi_arr - (en + 1.0) * pi0_arr

        # Flip p BEFORE backward scattering (matches numpy: p=-p then use)
        p = jnp.where(active, -p, p)

        # Scattering amplitudes
        ds1_1 = fn * (an_new * pi_arr + bn_new * tau_arr)
        ds2_1 = fn * (an_new * tau_arr + bn_new * pi_arr)
        ds1_2 = fn * p * (an_new * pi_arr - bn_new * tau_arr)
        ds2_2 = fn * p * (bn_new * pi_arr - an_new * tau_arr)

        s1_1 = jnp.where(active, s1_1 + ds1_1, s1_1)
        s1_2 = jnp.where(active, s1_2 + ds1_2, s1_2)
        s2_1 = jnp.where(active, s2_1 + ds2_1, s2_1)
        s2_2 = jnp.where(active, s2_2 + ds2_2, s2_2)

        # Update recurrence state (already frozen via psi/chi above)
        psi0 = jnp.where(active, psi1, psi0)
        psi1 = psi  # already gated
        chi0 = jnp.where(active, chi1, chi0)
        chi1 = chi  # already gated
        xi1 = psi1 - chi1 * 1j

        # pi recurrence
        pi1_new = ((2.0 * en + 1.0) * amu * pi_arr - (en + 1.0) * pi0_arr) / en
        pi0_arr = jnp.where(active, pi_arr, pi0_arr)
        pi1_arr = jnp.where(active, pi1_new, pi1_arr)

        return (psi0, psi1, chi0, chi1, xi1, qsca, gsca,
                an_new, bn_new, an_prev_new, bn_prev_new, p,
                s1_1, s1_2, s2_1, s2_2, pi0_arr, pi1_arr)

    result = jax.lax.fori_loop(0, NSTOP_MAX, mie_body, init_carry)
    (_, _, _, _, _, qsca, gsca,
     _, _, _, _, _,
     s1_1, s1_2, s2_1, s2_2, _, _) = result

    # Assemble full-range S1, S2
    S1 = jnp.concatenate([s1_1, jnp.flip(s1_2[:-1])])
    S2 = jnp.concatenate([s2_1, jnp.flip(s2_2[:-1])])

    gsca = 2.0 * gsca / qsca
    qsca = (2.0 / (dx * dx)) * qsca
    qext = (4.0 / (dx * dx)) * S1[0].real
    qback = 4.0 * (jnp.abs(S1[2 * nang - 2]) / dx)**2

    return S1, S2, qext, qsca, qback, gsca


@jax.jit
def bhmie_qsca_jax(x, refrel):
    """Lightweight Mie calculation — returns only (Qext, Qsca, gsca).

    Skips angular amplitude computation (S1, S2, pi, tau) for faster
    precomputation. JIT-compilable and vmappable.

    Args:
        x: Size parameter (scalar, traced).
        refrel: Complex refractive index (scalar, traced).

    Returns:
        (Qext, Qsca, gsca) tuple of scalars.
    """
    dx = x * 1.0
    y = x * refrel
    nstop = _compute_nstop(x)
    nmx = jnp.int32(jnp.maximum(x + 4.0 * x**0.3333 + 2.0,
                                  jnp.abs(y)) + 15.0)

    # --- Downward recurrence for logarithmic derivatives ---
    d = jnp.zeros(NMX_MAX, dtype=jnp.complex128)

    def recurrence_body(i, d):
        k = NMX_MAX - 2 - i
        en = jnp.float64(k + 2)
        new_val = (en / y) - (1.0 / (d[k + 1] + en / y))
        d = d.at[k].set(jnp.where(k < nmx - 1, new_val, 0.0 + 0j))
        return d

    d = jax.lax.fori_loop(0, NMX_MAX - 1, recurrence_body, d)

    # --- Main Mie series (no angular amplitudes) ---
    # carry: (psi0, psi1, chi0, chi1, xi1, qsca, gsca,
    #         an, bn, an_prev, bn_prev, s1_0_real)
    # s1_0_real tracks S1[0].real for Qext (forward scattering)
    psi0_init = jnp.cos(dx)
    psi1_init = jnp.sin(dx)
    chi0_init = -jnp.sin(dx)
    chi1_init = jnp.cos(dx)
    xi1_init = psi1_init - chi1_init * 1j
    zero_c = jnp.complex128(0.0)

    init_carry = (
        psi0_init, psi1_init, chi0_init, chi1_init, xi1_init,
        0.0, 0.0,                          # qsca, gsca
        zero_c, zero_c, zero_c, zero_c,    # an, bn, an_prev, bn_prev
        0.0,                                # s1_0_real (for Qext)
    )

    def mie_body(n, carry):
        (psi0, psi1, chi0, chi1, xi1, qsca, gsca,
         an, bn, an_prev, bn_prev, s1_0_real) = carry

        en = jnp.float64(n + 1)
        active = n < nstop

        # Riccati-Bessel recurrence — freeze when inactive to prevent overflow
        psi = jnp.where(active, (2.0 * en - 1.0) * psi1 / dx - psi0, psi1)
        chi = jnp.where(active, (2.0 * en - 1.0) * chi1 / dx - chi0, chi1)
        xi = psi - chi * 1j

        an_prev_new = an
        bn_prev_new = bn

        # Mie coefficients — guard denominators to prevent NaN in gradients
        d_n = d[jnp.clip(n, 0, NMX_MAX - 1)]
        numer_an = (d_n / refrel + en / dx) * psi - psi1
        denom_an = (d_n / refrel + en / dx) * xi - xi1
        numer_bn = (refrel * d_n + en / dx) * psi - psi1
        denom_bn = (refrel * d_n + en / dx) * xi - xi1
        safe_denom_an = jnp.where(active, denom_an, 1.0 + 0j)
        safe_denom_bn = jnp.where(active, denom_bn, 1.0 + 0j)
        an_new = jnp.where(active, numer_an / safe_denom_an, 0.0 + 0j)
        bn_new = jnp.where(active, numer_bn / safe_denom_bn, 0.0 + 0j)

        dq = (2.0 * en + 1.0) * (jnp.abs(an_new)**2 + jnp.abs(bn_new)**2)

        dg1 = ((2.0 * en + 1.0) / (en * (en + 1.0))) * (
            an_new.real * bn_new.real + an_new.imag * bn_new.imag)
        dg2 = jnp.where(
            n > 0,
            ((en - 1.0) * (en + 1.0) / en) * (
                an_prev_new.real * an_new.real + an_prev_new.imag * an_new.imag
                + bn_prev_new.real * bn_new.real + bn_prev_new.imag * bn_new.imag),
            0.0)

        qsca = jnp.where(active, qsca + dq, qsca)
        gsca = jnp.where(active, gsca + dg1 + dg2, gsca)

        # S1(0) for Qext: at theta=0, pi_n = tau_n = n(n+1)/2, so
        # fn * (an*pi + bn*tau) = fn * pi_n * (an+bn) = (2n+1)/2 * (an+bn)
        ds1_0 = ((2.0 * en + 1.0) / 2.0 * (an_new + bn_new)).real
        s1_0_real = jnp.where(active, s1_0_real + ds1_0, s1_0_real)

        psi0 = jnp.where(active, psi1, psi0)
        psi1 = psi  # already gated
        chi0 = jnp.where(active, chi1, chi0)
        chi1 = chi  # already gated
        xi1 = psi1 - chi1 * 1j

        return (psi0, psi1, chi0, chi1, xi1, qsca, gsca,
                an_new, bn_new, an_prev_new, bn_prev_new, s1_0_real)

    result = jax.lax.fori_loop(0, NSTOP_MAX, mie_body, init_carry)
    (_, _, _, _, _, qsca, gsca,
     _, _, _, _, s1_0_real) = result

    gsca = 2.0 * gsca / qsca
    qsca_out = (2.0 / (dx * dx)) * qsca
    qext = (4.0 / (dx * dx)) * s1_0_real

    return qext, qsca_out, gsca
