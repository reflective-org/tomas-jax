"""Bohren-Huffman Mie scattering calculation.

Compute scattering and extinction efficiencies for homogeneous spheres
using the exact Lorenz-Mie theory series expansion.

Ported from the classic Bohren & Huffman (1983) algorithm.
Reference implementation: references/radiative-forcing/bhmie.py

Reference:
    Bohren, C. F. and D. R. Huffman (1983), Absorption and Scattering
    of Light by Small Particles, Wiley-Interscience.
"""
import numpy as np


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
