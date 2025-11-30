# tomas-jax-coagulation ⚡

**High-Performance, Differentiable Aerosol Microphysics in JAX**

`tomas-jax-coagulation` is a modern re-implementation of the TOMAS (TwO-Moment Aerosol Sectional) microphysics algorithm, leveraging Google's JAX for:

- **Extreme Speed:** Just-In-Time (JIT) compilation via XLA for CPU, GPU, and TPU.
- **Automatic Differentiation:** Calculate gradients of the final state with respect to any input parameter (Temperature, Initial Concentration, Mode Diameter) "for free."
- **Modern Solvers:** Uses `diffrax` for adaptive, high-order ODE integration (`Tsit5`).
- **Vectorized Physics:** Full SIMD implementation of the TFL (Tzivion, Feingold, Levin) coagulation algorithm.

---

## 🚀 Features

- **Precision:** Enforced float64 everywhere for strict mass conservation ($< 10^{-13}$ relative error).
- **Operator Splitting:** Decouples expensive Kernel calculations ($O(N^2)$) from the integration loop for maximum efficiency.
- **MNFIX-JAX:** A fully vectorized, JIT-compatible implementation of the TOMAS mass-number drift correction algorithm.
- **Diagnostics:** Built-in plotting for Size Distributions, Banana Plots, and Time Series.

---

## 📦 Installation

**Requirements:** Python 3.9+

Clone the repository:

```bash
git clone https://github.com/your-username/tomas-jax-coagulation.git
cd tomas-jax-coagulation
```

Install in editable mode (installs package and dependencies: JAX, Diffrax, Matplotlib):

```bash
pip install -e .
```

> **Note:** If you have an NVIDIA GPU, ensure you install the CUDA-enabled JAX first (see [JAX documentation](https://github.com/google/jax#installation) for details).

---

## 🏃 Usage

**1. Standard Box Model Simulation**  
Runs a 0-D box model simulation for coagulation over time:

```bash
python run_box_model.py
```

_Outputs: Console logs + Plots (Banana Plot, Size Distribution, Rates)_

**2. Sensitivity Analysis (Automatic Differentiation)**  
Calculate $\frac{d(N_{\text{final}})}{d(\text{Concentration})}$:

```bash
python run_sensitivity_analysis.py
```

**3. Parameter Sweeps**  
Sensitivity to Temperature or Mode Diameter:

```bash
python run_temp_sensitivity.py
python run_gmd_sensitivity.py
```

---

## 📂 Project Structure

```
tomas-jax/
├── run_box_model.py             # Main Entry Point (Driver)
├── run_sensitivity_analysis.py  # AD Examples
├── pyproject.toml               # Build configuration
│
├── tomas_jax/                   # Source Package
│   ├── core/
│   │   ├── config.py            # Global Constants (Bins, Species)
│   │   ├── state.py             # JAX Data Structures
│   │   └── mnfix_jax.py         # Drift Correction Algorithm
│   │
│   ├── physics/
│   │   ├── coagulation_kernel.py # Brownian Coagulation (Fuchs)
│   │   ├── coagulation_rates.py  # TFL Coagulation Solver
│   │   ├── properties.py         # Diffusivity, Viscosity, Mean Free Path
│   │   └── density.py            # Aerosol Density (Tang 1997)
│   │
│   ├── solvers/
│   │   └── diffrax.py           # Adaptive ODE Integrator Wrapper
│   │
│   └── utils/
│       └── plotting.py          # Visualization Tools
```

---

## 🔬 Physics Details

### The Solver

- **Equation:** Multicomponent Stochastic Collection Equation (SCE)
- **Operator Splitting:** Coagulation Kernel $K_{ij}$ is pre-computed at the start of each model timestep (e.g., 60 s).
- **Integration:** `diffrax.Tsit5` (Runge-Kutta 5(4)) handles sub-stepping within each model timestep.
- **Correction:** MNFIX is periodically applied to fix bin drift from independent Number and Mass integration.

### The Kernel

- Brownian Coagulation Kernel with Fuchs correction for non-continuum effects (transition regime).

### Mass Conservation

- The TFL algorithm is conservative by design. However, floating-point drift can occur.
- Enforce float64 everywhere to keep relative mass error below $10^{-13}$.

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/nucleation`)
3. **Commit** your changes
4. **Push** to the branch
5. **Open** a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

_Built with [JAX](https://github.com/google/jax) and [Diffrax](https://github.com/patrick-kidger/diffrax)._
