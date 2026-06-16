"""Choice (c): geometry-free radiative-forcing *efficiency* for the Marianna case.

This is the box-honest RF metric. Two intensive quantities, both RATIOS of
(scattering) / (dry mass), so they depend ONLY on the *shape* of the size
distribution — not on the absolute amount. Consequences:
  - **dilution-invariant** (dilution scales Nk and Mk together; the ratio is
    unchanged), so the signal is purely the microphysical reshaping;
  - they quantify how good the aerosol is at scattering per unit mass, and how
    that degrades as coagulation grows particles off the ~0.3-0.5 µm optimum.

Quantities (per loaded NPZ time series):
  MSE     — mass scattering efficiency [m²/g]
  RF_eff  — RF per unit dry-mass burden [W/m² per g/m²]  (negative = cooling)

Optics follow the validated stratospheric setup: H₂SO₄/H₂O solution density
(so the mass→radius map gives the wet droplet radius), 500 nm, Pierce params.

Usage::
    python -m experimental_case.rf_efficiency --scenario 1
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tomas_jax.core.config import make_grid, XK0, ICOMP, SRTSO4, PI, AVOGADRO, MW_H2SO4

MW_S = 32.06   # g/mol, elemental sulfur
from tomas_jax.physics.radiative_forcing import (
    precompute_mie_properties, compute_rf_efficiency,
    compute_mass_scattering_efficiency,
    DENSITY_H2SO4_SOLUTION, REFINDEX_SULFATE,
)

# Stratospheric radiative parameters (Pierce et al. 2010 SI)
TATM_STRAT   = 1.0     # no overlying atmosphere above a stratospheric layer
ALBEDO_STRAT = 0.15    # clear-sky surface albedo
CLOUD_FRACTION = 0.6   # all-sky factor: multiply clear-sky efficiency by (1-this)
A_EARTH = 5.1e14       # m² (for global-mean RF contribution of one plume)

# RF target times (h). 2 wk requires a >= 336 h (14 d) run.
RF_TARGETS_H = [(48.0, '2 d'), (168.0, '1 wk'), (336.0, '2 wk')]

_MIE = {}
_ROPT = {}


def _get_mie(nbins):
    # Solar-broadband (spectrum-integrated) Mie — physically correct vs a single
    # wavelength. The forcing optimum is ~224 nm radius (Dp 448 nm) under this,
    # matching the literature/codebase ~400-440 nm RF diameter band.
    if nbins not in _MIE:
        from .run_marianna_dilution import make_grid_for
        xk = np.array(make_grid_for(nbins))
        _MIE[nbins] = precompute_mie_properties(
            xk, density=DENSITY_H2SO4_SOLUTION, refindex=REFINDEX_SULFATE,
            spectral=True, n_wavelengths=30)
    return _MIE[nbins]


def efficiency_timeseries(d):
    """Return (t_h, mse[m²/g], rf_eff[W/m² per g/m²]) for an NPZ dict."""
    nbins = int(d['nbins'])
    mie = _get_mie(nbins)
    Nk_all = np.asarray(d['Nk_every'])          # (nsteps, nbins) [#/cell]
    Mk_dry = np.asarray(d['Mk_dry_every'])      # (nsteps, nbins) [kg/cell] dry/bin

    # compute_*_efficiency only use Mk[:, :SRTH2O].sum -> put dry mass in SO4 col
    Mk_all = np.zeros((Nk_all.shape[0], nbins, ICOMP))
    Mk_all[:, :, SRTSO4] = Mk_dry
    Nk_j = jnp.asarray(Nk_all)
    Mk_j = jnp.asarray(Mk_all)

    mse = jax.vmap(lambda N, M: compute_mass_scattering_efficiency(N, M, mie)[0])(Nk_j, Mk_j)
    rfe = jax.vmap(lambda N, M: compute_rf_efficiency(
        N, M, mie, Tatm=TATM_STRAT, albedo=ALBEDO_STRAT))(Nk_j, Mk_j)
    return d['t_seconds'] / 3600.0, np.asarray(mse), np.asarray(rfe)


def optimum_radius_um():
    """Radius that maximizes per-mass forcing efficiency (∝ β·Q_sca/r),
    solar-broadband. Geometry-free. ~224 nm for wet sulfate (Dp ~448 nm),
    matching the literature/codebase ~400-440 nm RF diameter band.
    """
    if 'r' not in _ROPT:
        rfine = np.logspace(np.log10(1e-8), np.log10(8e-7), 200)
        m = (4.0 / 3.0) * PI * DENSITY_H2SO4_SOLUTION * rfine**3
        xk = np.concatenate([[m[0] * 0.9], np.sqrt(m[:-1] * m[1:]), [m[-1] * 1.1]])
        mf = precompute_mie_properties(
            xk, density=DENSITY_H2SO4_SOLUTION, refindex=REFINDEX_SULFATE,
            spectral=True, n_wavelengths=30)
        rr = np.asarray(mf.radii)
        per_mass = np.asarray(mf.upscatter_avg) * np.asarray(mf.Qsca) / rr
        _ROPT['r'] = float(rr[int(np.argmax(per_mass))])
    return _ROPT['r'] * 1e6


def effective_radius_um(d):
    """Effective (area-weighted) radius r_eff = Σ N r³ / Σ N r²  [µm], per time."""
    mie = _get_mie(int(d['nbins']))
    r = np.asarray(mie.radii)
    Nk = np.asarray(d['Nk_every'])
    reff = (Nk @ r**3) / np.maximum(Nk @ r**2, 1e-300)
    return reff * 1e6


def rf_global_timeseries(d):
    """Box-honest global-mean RF contribution of THIS plume [W/m², all-sky].

    RF_global(t) = efficiency[W/g] · M_plume(t)[g] / A_Earth, where
    M_plume = (dry-mass concentration) · V(t). All-sky (× (1-cloud_fraction)).
    Single plume — scales linearly with number of plumes / total injection.
    Also returns the total plume dry mass [kg].
    """
    _, _, rfe = efficiency_timeseries(d)
    eff_allsky = rfe * (1.0 - CLOUD_FRACTION)               # W/g
    Vt = np.asarray(d['v0_m3']) * np.asarray(d['V_ratio_every'])   # m³
    M_plume_g = np.asarray(d['M_dry_every']) * Vt * 1e3     # kg/m³ · m³ · g/kg
    rf_glob = eff_allsky * M_plume_g / A_EARTH              # W/m²
    return d['t_seconds'] / 3600.0, rf_glob, M_plume_g / 1e3


def sulfur_normalized(d):
    """Dilution-normalized sulfur metrics (per colleague's suggestion).

    Total sulfur [SO2 + gas H2SO4 + particulate SO4] is conserved by chemistry
    and lost only to dilution, so dividing by it cancels dilution exactly.

    Returns (t_h, conversion_fraction, forcing_per_S[W/g-S, all-sky]):
      - conversion_fraction = particulate-S / total-S  (gas→particle conversion)
      - forcing_per_S        = all-sky forcing power per gram of total sulfur
        = eff[W/g-aerosol] · (aerosol mass) / (total-S mass)
        ≈ dRF/dS in the optically-thin/linear regime.
    """
    _, _, rfe = efficiency_timeseries(d)
    eff_allsky = rfe * (1.0 - CLOUD_FRACTION)                  # W/g-aerosol
    SO2 = np.asarray(d['SO2_molec_cm3'])                       # gas SO2 [molec/cm³]
    H2SO4_gas = np.asarray(d['SO4_molec_cm3'])                 # gas H2SO4 [molec/cm³]
    Mdry = np.asarray(d['M_dry_every'])                        # kg/m³ (BOXVOL=1 m³)
    aer_molec_cm3 = (Mdry * 1e3 / MW_H2SO4) * AVOGADRO / 1e6   # particulate SO4 [molec/cm³]
    S_tot = SO2 + H2SO4_gas + aer_molec_cm3                    # total S [molec/cm³]
    f_conv = aer_molec_cm3 / np.maximum(S_tot, 1e-300)
    aer_g_cm3 = Mdry * 1e-3                                    # g-aerosol/cm³
    S_g_cm3 = S_tot * MW_S / AVOGADRO                          # g-S/cm³
    fps = eff_allsky * aer_g_cm3 / np.maximum(S_g_cm3, 1e-300) # W/g-S
    return d['t_seconds'] / 3600.0, f_conv, fps


def plot_sulfur_normalized(d, figdir):
    """Dilution-normalized: S conversion fraction and forcing per total sulfur."""
    from .plot_marianna_dilution import _sname, _despine
    t_h, f_conv, fps = sulfur_normalized(d)
    targets = _targets_in_range(t_h)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Sulfur-normalized (dilution-invariant)  •  {_sname(d)}',
                 fontsize=11, fontweight='bold')

    for ax, y, ylab, ttl in [
        (axes[0], f_conv * 100, 'particulate S / total S  [%]',
         'Gas→particle conversion'),
        (axes[1], fps, 'forcing per total S  [W/g-S, all-sky]',
         'Forcing per unit sulfur (≈ dRF/dS)'),
    ]:
        ax.plot(t_h, y, color='#1565C0', lw=2.0)
        for h, lab in targets:
            i = int(np.argmin(np.abs(t_h - h)))
            ax.axvline(h, color='#E53935', ls='--', lw=1.0, alpha=0.7)
            ax.annotate(f'{lab}\n{y[i]:.3g}', xy=(h, y[i]), xytext=(6, 0),
                        textcoords='offset points', fontsize=8, color='#B71C1C', va='center')
        ax.set_xlabel('Time [h]'); ax.set_ylabel(ylab); ax.set_title(ttl)
        _despine(ax); ax.grid(True, alpha=0.25, which='both')

    fig.tight_layout()
    out = os.path.join(figdir, 'sulfur_normalized.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)

    print('  Sulfur-normalized at target times:')
    print(f"    {'time':>6} | {'S conversion':>12} | {'forcing per S [W/g-S]':>22}")
    for h, lab in targets:
        i = int(np.argmin(np.abs(t_h - h)))
        print(f"    {lab:>6} | {f_conv[i]*100:>11.2f}% | {fps[i]:>22.4g}")
    return out


def _targets_in_range(t_h):
    t_end = t_h[-1]
    return [(h, lab) for h, lab in RF_TARGETS_H if h <= t_end + 1e-6]


# Pinatubo per-gram forcing reference (~20 Tg aerosol -> ~-3.5 W/m2 global)
PINATUBO_W_PER_G = -89.0


def plot_rf_efficiency(d, figdir):
    """Intensive, dilution-invariant forcing efficiency: MSE [m²/g] and the
    per-gram forcing W/g (all-sky). 'W/g' = forcing power per gram of aerosol
    (optically thin, global-mean insolation) — geometry-free, no burden needed.
    """
    from .plot_marianna_dilution import _sname, _despine
    t_h, mse, rfe = efficiency_timeseries(d)
    eff_wg = rfe * (1.0 - CLOUD_FRACTION)   # clear-sky -> all-sky W/g
    targets = _targets_in_range(t_h)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Forcing efficiency (intensive, dilution-invariant)  •  {_sname(d)}',
                 fontsize=11, fontweight='bold')

    for ax, y, ylab, ttl in [
        (axes[0], mse, 'MSE [m²/g]', 'Mass scattering efficiency'),
        (axes[1], eff_wg, 'forcing efficiency [W/g, all-sky]',
         'Per-gram forcing (negative = cooling)'),
    ]:
        ax.plot(t_h, y, color='#1565C0', lw=2.0)
        for h, lab in targets:
            i = int(np.argmin(np.abs(t_h - h)))
            ax.axvline(h, color='#E53935', ls='--', lw=1.0, alpha=0.7)
            ax.annotate(f'{lab}\n{y[i]:.3g}', xy=(h, y[i]),
                        xytext=(6, 0), textcoords='offset points',
                        fontsize=8, color='#B71C1C', va='center')
        ax.set_xlabel('Time [h]'); ax.set_ylabel(ylab); ax.set_title(ttl)
        _despine(ax); ax.grid(True, alpha=0.25, which='both')
    axes[1].axhline(PINATUBO_W_PER_G, color='#2E7D32', ls=':', lw=1.4,
                    label=f'Pinatubo ≈ {PINATUBO_W_PER_G:.0f} W/g')
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out = os.path.join(figdir, 'rf_efficiency.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)

    print('  Forcing efficiency (choice c) at target times:')
    print(f"    {'time':>6} | {'MSE [m²/g]':>12} | {'efficiency [W/g, all-sky]':>26}")
    for h, lab in targets:
        i = int(np.argmin(np.abs(t_h - h)))
        print(f"    {lab:>6} | {mse[i]:>12.4g} | {eff_wg[i]:>26.4g}")
    missing = [lab for h, lab in RF_TARGETS_H if h > t_h[-1] + 1e-6]
    if missing:
        print(f"    (not covered by this {t_h[-1]/24:.0f}-day run: {', '.join(missing)} "
              f"— extend max_hours to >= {max(h for h,_ in RF_TARGETS_H):.0f} h)")
    return out


def plot_rf_and_radius(d, figdir):
    """3-panel: RF(t) [global-mean, single plume], r_eff(t), and r_eff/r_opt(t)."""
    from .plot_marianna_dilution import _sname, _despine
    t_h, rf_glob, M_kg = rf_global_timeseries(d)
    reff = effective_radius_um(d)
    r_opt = optimum_radius_um()
    ratio = reff / r_opt
    targets = _targets_in_range(t_h)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'RF & size evolution  •  {_sname(d)}', fontsize=12, fontweight='bold')

    # 1. RF time series (global-mean contribution of this plume, all-sky)
    ax = axes[0]
    ax.plot(t_h, rf_glob, color='#1565C0', lw=2.0)
    ax.set_xlabel('Time [h]'); ax.set_ylabel('RF [W/m²]  (single plume, all-sky)')
    ax.set_title('Global-mean RF contribution\n(× N plumes for a campaign)')

    # 2. effective radius vs optimum
    ax = axes[1]
    ax.plot(t_h, reff * 1e3, color='#2E7D32', lw=2.0, label='r_eff')
    ax.axhline(r_opt * 1e3, color='#E53935', ls='--', lw=1.4,
               label=f'r_opt = {r_opt*1e3:.0f} nm')
    ax.set_xlabel('Time [h]'); ax.set_ylabel('effective radius [nm]')
    ax.set_title('Effective radius  r_eff = ⟨r³⟩/⟨r²⟩')
    ax.legend(frameon=False, fontsize=9)

    # 3. ratio r_eff / r_opt
    ax = axes[2]
    ax.plot(t_h, ratio, color='#6A1B9A', lw=2.0)
    ax.axhline(1.0, color='#E53935', ls='--', lw=1.4, label='optimum (=1)')
    ax.fill_between(t_h, ratio, 1.0, where=(ratio < 1.0), color='#6A1B9A', alpha=0.10)
    ax.set_xlabel('Time [h]'); ax.set_ylabel('r_eff / r_opt')
    ax.set_title('Distance from optimal scattering size\n(<1 = undersized)')
    ax.legend(frameon=False, fontsize=9)

    for ax in axes:
        for h, lab in targets:
            ax.axvline(h, color='#999999', ls=':', lw=0.9, alpha=0.7)
        _despine(ax); ax.grid(True, alpha=0.25, which='both')

    fig.tight_layout()
    out = os.path.join(figdir, 'rf_and_radius.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)

    print('  RF / radius at target times:')
    print(f"    {'time':>6} | {'RF[W/m²]':>11} | {'M_plume[kg]':>11} | {'r_eff[nm]':>9} | {'r_eff/r_opt':>11}")
    for h, lab in targets:
        i = int(np.argmin(np.abs(t_h - h)))
        print(f"    {lab:>6} | {rf_glob[i]:>11.2e} | {M_kg[i]:>11.1f} | "
              f"{reff[i]*1e3:>9.1f} | {ratio[i]:>11.2f}")
    return out


if __name__ == '__main__':
    import argparse
    from .run_marianna_dilution import SCENARIOS
    from .plot_marianna_dilution import _load
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='B1-D2', choices=sorted(SCENARIOS))
    ap.add_argument('--npz', default=None)
    args = ap.parse_args()
    npz = args.npz or SCENARIOS[args.scenario].npz
    d = _load(npz)
    figdir = os.path.join(os.path.dirname(npz), 'figs')
    os.makedirs(figdir, exist_ok=True)
    print(f'  Saved: {plot_rf_efficiency(d, figdir)}')
    print(f'  Saved: {plot_rf_and_radius(d, figdir)}')
