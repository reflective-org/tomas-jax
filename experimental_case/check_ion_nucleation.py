"""Pre-check: does ion-induced nucleation actually fire under the Marianna
conditions (T=210 K, P=55 hPa, NH3=0, ion-pair production = 30 pairs/cm³/s)?

The Marianna case uses the ``ricco_dunne`` scheme with organics disabled and
NH3=0, so only the Dunne (2016) inorganic mechanisms are active:
    Jbn  binary neutral
    Jtn  ternary neutral   (-> 0 when NH3=0)
    Jbi  binary ion        (requires fion>0)
    Jti  ternary ion       (-> 0 when NH3=0)

This script sweeps gas-phase H2SO4 and prints, for fion=0 vs fion=30:
  - the steady-state small-ion concentration,
  - the per-mechanism rates,
  - the total rate,
confirming (a) Jbi>0 when fion>0, and (b) total(fion=30) > total(fion=0).

If this assertion fails, ion nucleation is NOT working and the full run
should not proceed.

Usage::
    python -m experimental_case.check_ion_nucleation
"""
import numpy as np
import jax.numpy as jnp

from tomas_jax.physics.nucleation import dunne_nucleation_rate, _compute_ionc

# Marianna ambient conditions
TEMP = 210.0          # K
PRES = 5500.0         # Pa
NH3  = 0.0            # molec/cm³  (no ammonia)
FION_ON  = 30.0       # pairs/cm³/s
FION_OFF = 0.0

# Air number density [molec/cm³] (same formula as ricco_dunne_nucleation_rate)
MAIR = 2.69e19 * 273.15 / TEMP * PRES / 101325.0

# H2SO4 sweep [molec/cm³]
H2SO4_SWEEP = np.logspace(6, 9, 7)


def _rates(fion, h2so4):
    fn, Jbn, Jtn, Jbi, Jti = dunne_nucleation_rate(
        jnp.float64(TEMP), jnp.float64(fion),
        jnp.float64(h2so4), jnp.float64(NH3), jnp.float64(MAIR))
    return (float(fn), float(Jbn), float(Jtn), float(Jbi), float(Jti))


def main():
    print('=' * 78)
    print('Ion-nucleation pre-check  (Dunne 2016, ricco_dunne scheme)')
    print(f'  T={TEMP} K, P={PRES/100:.0f} hPa, NH3={NH3}, Mair={MAIR:.3e} molec/cm³')
    print('=' * 78)

    ionc_on  = float(_compute_ionc(jnp.float64(FION_ON),  jnp.float64(TEMP), jnp.float64(MAIR)))
    ionc_off = float(_compute_ionc(jnp.float64(FION_OFF), jnp.float64(TEMP), jnp.float64(MAIR)))
    print(f'  Steady-state ion conc: fion=30 -> {ionc_on:.3e} cm⁻³   '
          f'fion=0 -> {ionc_off:.3e} cm⁻³')
    print('-' * 78)
    hdr = (f"{'H2SO4':>10} | {'J_tot(0)':>11} {'J_tot(30)':>11} "
           f"{'Jbn':>11} {'Jbi(30)':>11} {'ratio':>7}")
    print(hdr)
    print('-' * 78)

    all_ok = True
    any_active = False
    for h2so4 in H2SO4_SWEEP:
        fn0, Jbn0, _, Jbi0, _ = _rates(FION_OFF, h2so4)
        fn1, Jbn1, _, Jbi1, _ = _rates(FION_ON, h2so4)
        ratio = fn1 / fn0 if fn0 > 0 else np.inf
        print(f"{h2so4:>10.1e} | {fn0:>11.3e} {fn1:>11.3e} "
              f"{Jbn1:>11.3e} {Jbi1:>11.3e} {ratio:>7.2f}")
        # Assertions
        if not (Jbi1 > 0.0):
            all_ok = False
        if not (fn1 >= fn0):
            all_ok = False
        if fn1 > fn0 * 1.0000001:
            any_active = True
        # Jbi must vanish when fion=0
        if Jbi0 != 0.0:
            all_ok = False

    print('-' * 78)
    print(f"  Jbi(fion=0) == 0 everywhere      : {'PASS' if all_ok else 'FAIL'}")
    print(f"  Jbi(fion=30) > 0                 : {'PASS' if all_ok else 'FAIL'}")
    print(f"  total(fion=30) > total(fion=0)   : {'PASS' if any_active else 'FAIL'}")
    print('=' * 78)

    if not (all_ok and any_active):
        raise SystemExit(
            'ION NUCLEATION CHECK FAILED — ion mechanism not contributing. '
            'Do not proceed with the full run until resolved.')
    print('Ion nucleation is ACTIVE. Safe to proceed with fion=30.')


if __name__ == '__main__':
    main()
