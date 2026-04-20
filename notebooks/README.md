# Notebooks

Educational Jupyter notebooks for the TOMAS-JAX aerosol microphysics model.

## Contents

- `radiative_forcing_tutorial.ipynb` — Interactive tutorial on stratospheric
  aerosol direct shortwave radiative forcing (Pierce et al. 2010). Teaches the
  Chylek & Wong (1995) equation, Mie scattering, upscatter fraction, and
  H2SO4/H2O composition. Ends with an interactive widget that recomputes the
  scattering-efficiency-vs-radius curve as the user moves sliders.

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
