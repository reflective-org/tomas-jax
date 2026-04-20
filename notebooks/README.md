# Notebooks

Educational Jupyter notebooks for the TOMAS-JAX aerosol microphysics model.

## Contents

- `radiative_forcing_tutorial.ipynb` — Interactive educational tutorial on
  stratospheric aerosol direct shortwave radiative forcing, following the
  simple-model formulation of Pierce et al. (2010). Covers Mie scattering,
  upscatter fraction (Wiscombe & Grams 1976), H2SO4/H2O equilibrium
  composition (Tabazadeh et al. 1997), single-wavelength vs full-spectrum
  Mie, and reproduces Pierce (2010) Figure 1. Ends with an interactive
  playground where researchers can dial in temperature, humidity,
  refractive index, and atmospheric parameters and see the cooling
  efficiency curve update live.

## Launching

Install the notebook extras:

```bash
uv sync --extra notebook
```

Then start JupyterLab:

```bash
jupyter lab notebooks/
```

## Platforms

These notebooks are designed to run on:

- Local JupyterLab
- CloudHub (a researcher-facing JupyterHub climate-analysis platform)
- Any environment with `tomas_jax`, `ipywidgets`, and `matplotlib` installed
