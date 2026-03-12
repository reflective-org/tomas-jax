"""Validate SO2+OH chemistry against Sun et al. (2022) figures.

Generates 6 figures:
  A — k1 vs [M] at 5 temperatures (cf. Fig. 9)
  B — k1 vs [M] at 298K: our code vs IUPAC/NASA (cf. Fig. 10)
  C — Fall-off for N2 vs H2O bath gases (cf. Fig. 12)
  D — k1 vs altitude (standard atmosphere)
  E — SO2 lifetime vs [OH]
  F — H2O enhancement map (cf. Fig. 13 concept)

Usage:
    python -m benchmarks.python.validate_so2_chemistry
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tomas_jax.physics.so2_chemistry import calc_k1_so2_oh, _FC
from tomas_jax.core.config import KB


# =========================================================================
# Helper: IUPAC and NASA rate constants for comparison
# =========================================================================

def k1_iupac(T, M):
    """IUPAC (2021) recommendation for SO2+OH+M.

    k0 = 3.3e-31 (T/300)^-4.3
    k_inf = 1.6e-12
    Fc = 0.6
    """
    k0 = 3.3e-31 * (T / 300.0) ** (-4.3)
    kinf = 1.6e-12
    Fc = 0.6
    ratio = k0 * M / kinf
    N = 0.75 - 1.27 * np.log10(Fc)
    log10_F = np.log10(Fc) / (1.0 + (np.log10(np.maximum(ratio, 1e-300)) / N) ** 2)
    F = 10.0 ** log10_F
    return ratio / (1.0 + ratio) * kinf * F


def k1_nasa(T, M):
    """NASA/JPL (2020) recommendation for SO2+OH+M.

    k0 = 3.3e-31 (T/300)^-4.3
    k_inf = 1.6e-12
    Fc = 0.6
    (Same as IUPAC for this reaction)
    """
    return k1_iupac(T, M)


# =========================================================================
# Output directory
# =========================================================================
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'so2_chemistry')


def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)


# =========================================================================
# Figure A: k1 vs [M] at 5 temperatures
# =========================================================================

def figure_a():
    """Reproduce Sun et al. Fig. 9: k1 fall-off curves at 5 temperatures."""
    M_range = np.logspace(16, 20, 200)  # molec/cm^3
    temps = [220, 250, 273, 298, 333]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(8, 6))

    for T, c in zip(temps, colors):
        # Convert [M] to pressure for our function: p = [M] * kB * T / 1e-6
        pres_arr = M_range * KB * T / 1e-6  # Pa
        k1_vals = np.array([float(calc_k1_so2_oh(float(T), float(p))) for p in pres_arr])
        ax.loglog(M_range, k1_vals, color=c, label=f'{T} K', linewidth=2)

    ax.set_xlabel('[M] (molec cm$^{-3}$)', fontsize=12)
    ax.set_ylabel('k$_1$ (cm$^3$ molec$^{-1}$ s$^{-1}$)', fontsize=12)
    ax.set_title('SO$_2$ + OH + M: Fall-off curves (Sun et al. 2022, Method 1)', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(1e16, 1e20)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig_a_k1_vs_M_temperatures.png'), dpi=150)
    plt.close(fig)
    print("  Figure A saved.")


# =========================================================================
# Figure B: k1 vs [M] at 298K — comparison with IUPAC/NASA
# =========================================================================

def figure_b():
    """Reproduce Sun et al. Fig. 10: comparison at 298K."""
    M_range = np.logspace(16, 20, 200)
    T = 298.0

    pres_arr = M_range * KB * T / 1e-6
    k1_sun = np.array([float(calc_k1_so2_oh(T, float(p))) for p in pres_arr])
    k1_iup = k1_iupac(T, M_range)
    k1_nas = k1_nasa(T, M_range)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(M_range, k1_sun, 'r-', label='Sun et al. 2022 (Method 1)', linewidth=2)
    ax.loglog(M_range, k1_iup, 'b--', label='IUPAC 2021', linewidth=1.5)
    ax.loglog(M_range, k1_nas, 'g:', label='NASA/JPL 2020', linewidth=1.5)

    # Mark 1 atm
    M_1atm = 101325.0 / (KB * T) * 1e-6
    ax.axvline(M_1atm, color='gray', linestyle=':', alpha=0.5)
    ax.text(M_1atm * 1.1, 1e-13, '1 atm', fontsize=9, color='gray')

    ax.set_xlabel('[M] (molec cm$^{-3}$)', fontsize=12)
    ax.set_ylabel('k$_1$ (cm$^3$ molec$^{-1}$ s$^{-1}$)', fontsize=12)
    ax.set_title('SO$_2$ + OH + M at 298 K: Sun vs IUPAC vs NASA', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(1e16, 1e20)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig_b_k1_comparison_298K.png'), dpi=150)
    plt.close(fig)
    print("  Figure B saved.")


# =========================================================================
# Figure C: N2 vs H2O bath gas fall-off
# =========================================================================

def figure_c():
    """Sun et al. Fig. 12 concept: fall-off for N2 vs H2O bath gases."""
    M_range = np.logspace(16, 20, 200)
    T = 298.0

    # Pure N2 (rh=0)
    pres_arr = M_range * KB * T / 1e-6
    k1_n2 = np.array([float(calc_k1_so2_oh(T, float(p), rh=0.0)) for p in pres_arr])

    # Simulate "pure H2O" by computing k1 with H2O-only k0
    # k0_H2O = 1.65e-30 (T/300)^-4.90
    T_ratio = T / 300.0
    k0_h2o = 1.65e-30 * T_ratio ** (-4.90)
    kinf = 2.00e-12
    Fc = _FC
    N = 0.75 - 1.27 * np.log10(Fc)

    ratio_h2o = k0_h2o * M_range / kinf
    log10_F = np.log10(Fc) / (1.0 + (np.log10(np.maximum(ratio_h2o, 1e-300)) / N) ** 2)
    F_h2o = 10.0 ** log10_F
    k1_h2o = ratio_h2o / (1.0 + ratio_h2o) * kinf * F_h2o

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(M_range / 1e19, k1_n2 / 1e-12, 'b-', label='N$_2$ bath gas', linewidth=2)
    ax.plot(M_range / 1e19, k1_h2o / 1e-12, 'r-', label='H$_2$O bath gas', linewidth=2)

    ax.set_xlabel('[M] ($\\times 10^{19}$ molec cm$^{-3}$)', fontsize=12)
    ax.set_ylabel('k$_1$ ($\\times 10^{-12}$ cm$^3$ molec$^{-1}$ s$^{-1}$)', fontsize=12)
    ax.set_title('SO$_2$ + OH: N$_2$ vs H$_2$O bath gas at 298 K', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig_c_n2_vs_h2o_bath.png'), dpi=150)
    plt.close(fig)
    print("  Figure C saved.")


# =========================================================================
# Figure D: k1 vs altitude (standard atmosphere)
# =========================================================================

def figure_d():
    """k1 vs altitude using US Standard Atmosphere 1976."""
    altitudes = np.linspace(0, 30, 100)  # km

    # US Standard Atmosphere approximation
    T0, P0, L = 288.15, 101325.0, 0.0065  # K, Pa, K/m
    T_trop = 216.65  # K (tropopause)
    P_trop = P0 * (T_trop / T0) ** (9.80665 / (L * 287.05))

    temps = np.zeros_like(altitudes)
    pres = np.zeros_like(altitudes)
    for i, z in enumerate(altitudes):
        z_m = z * 1000.0
        if z_m < 11000:
            temps[i] = T0 - L * z_m
            pres[i] = P0 * (temps[i] / T0) ** (9.80665 / (L * 287.05))
        else:
            temps[i] = T_trop
            pres[i] = P_trop * np.exp(-9.80665 * (z_m - 11000) / (287.05 * T_trop))

    k1_sun = np.array([float(calc_k1_so2_oh(float(t), float(p)))
                        for t, p in zip(temps, pres)])

    M_arr = pres / (KB * temps) * 1e-6
    k1_iup = k1_iupac(temps, M_arr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(k1_sun * 1e12, altitudes, 'r-', label='Sun et al. 2022', linewidth=2)
    ax1.plot(k1_iup * 1e12, altitudes, 'b--', label='IUPAC 2021', linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('k$_1$ ($\\times 10^{-12}$ cm$^3$ molec$^{-1}$ s$^{-1}$)', fontsize=12)
    ax1.set_ylabel('Altitude (km)', fontsize=12)
    ax1.set_title('k$_1$(SO$_2$+OH) vs Altitude', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Temperature and pressure profiles
    ax2.plot(temps, altitudes, 'r-', label='T [K]', linewidth=1.5)
    ax2b = ax2.twiny()
    ax2b.plot(pres / 100, altitudes, 'b--', label='P [hPa]', linewidth=1.5)
    ax2.set_xlabel('Temperature (K)', fontsize=12, color='r')
    ax2b.set_xlabel('Pressure (hPa)', fontsize=12, color='b')
    ax2.set_ylabel('Altitude (km)', fontsize=12)
    ax2.set_title('Standard Atmosphere', fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig_d_k1_vs_altitude.png'), dpi=150)
    plt.close(fig)
    print("  Figure D saved.")


# =========================================================================
# Figure E: SO2 lifetime vs [OH]
# =========================================================================

def figure_e():
    """SO2 lifetime vs OH concentration."""
    oh_range = np.logspace(5, 7, 100)
    k1_298 = float(calc_k1_so2_oh(298.0, 101325.0))

    tau_days = 1.0 / (k1_298 * oh_range) / 86400.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(oh_range, tau_days, 'r-', linewidth=2)

    # Mark [OH] = 1e6 reference
    tau_ref = 1.0 / (k1_298 * 1e6) / 86400.0
    ax.plot(1e6, tau_ref, 'ko', markersize=8)
    ax.annotate(f'[OH]=10$^6$: {tau_ref:.1f} days',
                xy=(1e6, tau_ref), xytext=(2e6, tau_ref * 2),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('[OH] (molec cm$^{-3}$)', fontsize=12)
    ax.set_ylabel('SO$_2$ lifetime (days)', fontsize=12)
    ax.set_title(f'SO$_2$ lifetime at 298 K, 1 atm (k$_1$ = {k1_298:.2e})', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig_e_so2_lifetime.png'), dpi=150)
    plt.close(fig)
    print("  Figure E saved.")


# =========================================================================
# Figure F: H2O enhancement map
# =========================================================================

def figure_f():
    """H2O enhancement of k1: fractional change when adding water vapor."""
    temps = np.linspace(220, 320, 50)
    rh_values = [0.1, 0.3, 0.5, 0.8]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(8, 5))

    for rh, c in zip(rh_values, colors):
        enhancement = np.zeros_like(temps)
        for i, T in enumerate(temps):
            k1_dry = float(calc_k1_so2_oh(float(T), 101325.0, 0.0))
            k1_wet = float(calc_k1_so2_oh(float(T), 101325.0, float(rh)))
            enhancement[i] = (k1_wet - k1_dry) / k1_dry * 100.0

        ax.plot(temps, enhancement, color=c, label=f'RH = {int(rh*100)}%', linewidth=2)

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('k$_1$ enhancement (%)', fontsize=12)
    ax.set_title('H$_2$O enhancement of SO$_2$+OH rate at 1 atm', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig_f_h2o_enhancement.png'), dpi=150)
    plt.close(fig)
    print("  Figure F saved.")


# =========================================================================
# Main
# =========================================================================

def main():
    ensure_outdir()
    print(f"Generating SO2 chemistry validation figures in {OUTDIR}/")
    print()

    # Print reference k1 value
    k1_ref = float(calc_k1_so2_oh(298.0, 101325.0))
    tau_ref = 1.0 / (k1_ref * 1e6) / 86400.0
    print(f"  k1(298K, 1atm) = {k1_ref:.3e} cm^3/molec/s")
    print(f"  SO2 lifetime at [OH]=1e6: {tau_ref:.1f} days")
    print()

    figure_a()
    figure_b()
    figure_c()
    figure_d()
    figure_e()
    figure_f()

    print(f"\nAll 6 figures saved to {OUTDIR}/")


if __name__ == '__main__':
    main()
