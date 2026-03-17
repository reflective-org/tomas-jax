# Dilution / Entrainment

## Physics

Dilution is implemented as **box volume expansion**. As the box mixes with clean or ambient air, the volume grows while total per-cell mass is conserved (for clean air):

```
dV/dt = kdil * V

V(t+dt) = V(t) * exp(kdil * dt)
```

Concentrations decrease naturally as the inverse of volume:

```
C(t) = N_per_cell / V(t)
     = C(0) * V(0)/V(t)
     = C(0) * exp(-kdil * t)
```

This is physically correct for TOMAS, where Nk [#/cell], Mk [kg/cell], and Gc [kg/cell] are **total amounts** in the box, not concentrations per volume.

For **ambient background** entrainment, the entrained air brings in mass at the background concentration. The extra volume `dV = V(t+dt) - V(t)` carries:

```
Nk_new = Nk + Nk_bg_conc * dV    [#/cm³ × cm³ = #]
Mk_new = Mk + Mk_bg_conc * dV    [kg/cm³ × cm³ = kg]
Gc_new = Gc + Gc_bg_conc * dV    [kg/cm³ × cm³ = kg]
```

**Reference:** TRACER_SOM-TOMAS `box.f` — `BOXVOL = initial_boxvol * dilt_rate_new`

## Parameters

| Parameter | Units | Description |
|-----------|-------|-------------|
| `kdil` | s⁻¹ | Dilution rate (volume expansion rate) |
| `Nk_bg_conc` | #/cm³ | Background number concentration (0 for clean air) |
| `Mk_bg_conc` | kg/cm³ | Background mass concentration (0 for clean air) |
| `Gc_bg_conc` | kg/cm³ | Background gas concentration (0 for clean air) |

### Typical kdil values

- **Boundary layer growth**: kdil ~ 1e-4 s⁻¹ (tau ~ 2.8 h)
- **Chamber ventilation**: kdil ~ 1e-5 to 1e-4 s⁻¹ (tau ~ 2.8–28 h)
- **Plume dilution**: kdil ~ 1e-3 s⁻¹ (tau ~ 17 min)

## Implementation

- **File**: `tomas_jax/physics/dilution.py`
- **Function**: `dilution_step(boxvol, Nk, Mk, Gc, dt, kdil, Nk_bg_conc, Mk_bg_conc, Gc_bg_conc)`
- **Returns**: `(boxvol_new, Nk_new, Mk_new, Gc_new)`
- **JIT-compatible**: Yes
- **Integration**: Available as `'dilution'` process in `make_step()`

### Mass conservation

- **Clean air** (Cbg = 0): Per-cell totals Nk, Mk, Gc are exactly preserved. Mass conservation error should be ~0.
- **Ambient background**: Per-cell totals increase (mass entrained). Mass conservation is not defined (open system with mass injection).

### Effect on other processes

When boxvol changes, all processes that depend on concentration (condensation, coagulation, nucleation, SO2 chemistry) naturally use the updated boxvol:

- **Condensation**: Gas concentration = Gc/boxvol decreases → condensation slows
- **Coagulation**: N/boxvol decreases → coagulation rate (∝ n²) decreases
- **Nucleation**: Gas concentrations decrease → nucleation rate decreases

This is physically correct — diluted air has lower concentrations, so all rate-dependent processes slow down.

## Usage

### Via make_step (composable API)

```python
step = make_step(['coagulation', 'condensation', 'dilution'], cond_method='ppm_jit')

# Clean-air dilution (boxvol grows, mass conserved):
Nk, Mk, Gc, boxvol = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                            kdil=1e-4)

# Ambient background (entrained air adds mass):
Nk_bg_conc = Nk_init / boxvol_init  # [#/cm³]
Mk_bg_conc = Mk_init / boxvol_init  # [kg/cm³]
Gc_bg_conc = Gc_init / boxvol_init  # [kg/cm³]
Nk, Mk, Gc, boxvol = step(..., kdil=1e-4,
                            Nk_bg_conc=Nk_bg_conc,
                            Mk_bg_conc=Mk_bg_conc,
                            Gc_bg_conc=Gc_bg_conc)
```

Note: `make_step()` always returns 4 values `(Nk, Mk, Gc, boxvol)`. When dilution is not active, boxvol is returned unchanged.

### Via CLI

```bash
python run_box_model.py --make-step --dilution-rate 1e-4                    # clean-air
python run_box_model.py --make-step --dilution-rate 1e-4 --dilution-bg ambient  # entrained
```

### Direct function call

```python
from tomas_jax.physics.dilution import dilution_step

boxvol_new, Nk_new, Mk_new, Gc_new = dilution_step(
    boxvol, Nk, Mk, Gc, dt=60.0, kdil=1e-4)
```

## Passive Tracer

A passive scalar tracer tracks the concentration decrease due to volume expansion:

```python
from tomas_jax.physics.dilution import dilute_tracer

tracer = 1.0
for step in range(nsteps):
    tracer = dilute_tracer(tracer, dt, kdil)
# tracer == exp(-kdil * t) == V(0)/V(t)
```

The tracer matches the inverse of the volume expansion ratio, providing a reference to isolate dilution from chemistry/microphysics.

---

*Fortran reference: TRACER_SOM-TOMAS `box.f` (volume-based dilution)*
