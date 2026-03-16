# Dilution / Entrainment

## Physics

First-order relaxation of all concentrations toward a background state:

```
dC/dt = -kdil * (C - Cbg)
```

Analytical solution (exact for constant kdil and Cbg over dt):

```
C(t+dt) = Cbg + (C(t) - Cbg) * exp(-kdil * dt)
```

When Cbg = 0 (clean-air dilution): `C(t+dt) = C(t) * exp(-kdil * dt)`.

Applies independently to Nk (number per bin), Mk (mass per bin per species), and Gc (gas concentrations).

## Parameters

| Parameter | Units | Typical Values |
|-----------|-------|----------------|
| `kdil` | s⁻¹ | 1e-5 to 1e-3 |
| `Nk_bg` | #/grid cell | 0 (clean air) or background distribution |
| `Mk_bg` | kg/grid cell | 0 (clean air) or background distribution |
| `Gc_bg` | kg/grid cell | 0 (clean air) or background gas concentrations |

### Typical kdil values

- **Boundary layer growth**: kdil ~ 1e-4 s⁻¹ (tau ~ 2.8 h)
- **Chamber ventilation**: kdil ~ 1e-5 to 1e-4 s⁻¹ (tau ~ 2.8–28 h)
- **Plume dilution**: kdil ~ 1e-3 s⁻¹ (tau ~ 17 min)

## Implementation

- **File**: `tomas_jax/physics/dilution.py`
- **Function**: `dilution_step(Nk, Mk, Gc, dt, kdil, Nk_bg, Mk_bg, Gc_bg)`
- **JIT-compatible**: Yes
- **Integration**: Available as `'dilution'` process in `make_step()`

## Usage

### Via make_step (composable API)

```python
step = make_step(['coagulation', 'condensation', 'dilution'], cond_method='ppm_jit')
Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                    kdil=1e-4)  # clean-air dilution (Cbg = 0)
```

### Via CLI

```bash
python run_box_model.py --make-step --dilution-rate 1e-4                    # clean-air
python run_box_model.py --make-step --dilution-rate 1e-4 --dilution-bg ambient  # relax to initial
```

### Direct function call

```python
from tomas_jax.physics.dilution import dilution_step
Nk_new, Mk_new, Gc_new = dilution_step(Nk, Mk, Gc, dt=60.0, kdil=1e-4)
```

## Passive Tracer

A passive scalar tracer is available to isolate the dilution signal from reactive species. The tracer starts at 1.0 and decays purely by dilution — no chemistry, deposition, or microphysics.

```python
from tomas_jax.physics.dilution import dilute_tracer

tracer = 1.0
for step in range(nsteps):
    tracer = dilute_tracer(tracer, dt, kdil)
# tracer == exp(-kdil * t) for clean-air dilution
```

In the box model, the tracer is tracked automatically when `--dilution-rate > 0` and printed alongside `N_total` ratio at the end, allowing direct comparison of how much concentration change is due to dilution alone vs. chemistry/microphysics.

---

*Fortran reference: `layer_diff.f` (height-dependent layer diffusion)*
