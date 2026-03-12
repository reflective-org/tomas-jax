# TOMAS-JAX Web Interface — Architecture & Implementation Specification

**Date:** 2026-03-10
**Status:** Planning
**Author:** Generated with Claude Code

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Backend API (tomas-api)](#4-backend-api-tomas-api)
5. [Frontend Application (tomas-web)](#5-frontend-application-tomas-web)
6. [Data Flow](#6-data-flow)
7. [Deployment](#7-deployment)
8. [Security](#8-security)
9. [Performance & Scalability](#9-performance--scalability)
10. [Cost Estimation](#10-cost-estimation)
11. [Development Roadmap](#11-development-roadmap)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Appendix: API Reference](#appendix-a-api-reference)
14. [Appendix: Parameter Ranges](#appendix-b-parameter-ranges)

---

## 1. Overview

### Goal

Provide a public web interface where atmospheric scientists can run TOMAS aerosol microphysics simulations (coagulation, condensation, nucleation) in a browser — no local Python/JAX installation required.

### Target Users

- Atmospheric scientists and graduate students
- Researchers evaluating TOMAS for their models
- Educators teaching aerosol microphysics

### Key Capabilities

| Feature | Description |
|---------|-------------|
| Box model simulation | 24-hour aerosol evolution with configurable parameters |
| Process selection | Enable/disable nucleation, coagulation, condensation independently |
| Method selection | TFL vs PPM condensation, Tsit5 vs Euler coagulation |
| Interactive visualization | Size distributions, banana plots, time series, conservation diagnostics |
| Preset scenarios | Quick-start with validated parameter sets from the 50-scenario benchmark |
| Result download | Export simulation results as CSV/NPZ for offline analysis |

---

## 2. Architecture

### Two-Repository Pattern

```
┌────────────────────────────────────────────────────────────────────┐
│                          Public Internet                           │
└────────────────────────────┬───────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼──────────┐       ┌──────────▼─────────┐
    │   tomas-web (CDN)  │       │   tomas-api (VM)   │
    │                    │       │                    │
    │  React SPA         │◄─────►│  FastAPI + JAX     │
    │  Plotly.js charts  │ HTTPS │  Task queue        │
    │  Static hosting    │       │  Result cache      │
    │                    │       │                    │
    │  Vercel / Netlify  │       │  GCP / AWS / Fly   │
    └────────────────────┘       └────────────────────┘
```

**Why two repos:**

1. **Independent deployment** — Frontend deploys in seconds (static CDN), backend needs Python/JAX environment
2. **Scaling** — Frontend scales automatically on CDN; backend can be scaled separately based on compute demand
3. **Technology mismatch** — JS/React toolchain vs Python/JAX toolchain; separate CI/CD pipelines
4. **Cost** — Frontend hosting is free (Vercel/Netlify); backend compute is the real cost

**When a monorepo makes sense:** If only one person maintains both and you want simpler git history, use a monorepo with `api/` and `web/` directories. The architecture below works either way.

---

## 3. Repository Structure

### `tomas-api/` — Backend

```
tomas-api/
├── app/
│   ├── main.py                  # FastAPI application, CORS, lifespan
│   ├── config.py                # Environment variables, limits
│   ├── routers/
│   │   ├── simulate.py          # POST /simulate — submit simulation
│   │   ├── results.py           # GET /results/{job_id} — poll/fetch results
│   │   ├── presets.py           # GET /presets — preset scenarios
│   │   └── health.py            # GET /health — status + warmup state
│   ├── services/
│   │   ├── simulation.py        # Core: wraps tomas_jax run_box_model logic
│   │   ├── warmup.py            # JIT warmup on startup
│   │   └── job_manager.py       # In-memory job queue + result cache
│   ├── models/
│   │   ├── request.py           # Pydantic models for simulation input
│   │   ├── response.py          # Pydantic models for simulation output
│   │   └── enums.py             # CondMethod, CoagSolver, ProcessMode enums
│   └── utils/
│       ├── units.py             # Unit conversions (molec/cm³ ↔ kg/cell)
│       └── validation.py        # Physical plausibility checks
├── tests/
│   ├── test_simulate.py
│   ├── test_validation.py
│   └── test_presets.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt             # fastapi, uvicorn, jax[cpu], tomas-jax
├── pyproject.toml
└── README.md
```

### `tomas-web/` — Frontend

```
tomas-web/
├── public/
│   └── index.html
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── api/
│   │   └── client.ts            # API client (fetch wrapper, types)
│   ├── components/
│   │   ├── Layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Footer.tsx
│   │   │   └── Sidebar.tsx
│   │   ├── SimulationForm/
│   │   │   ├── SimulationForm.tsx      # Main form container
│   │   │   ├── AerosolParams.tsx       # N_total, GMD, GSD inputs
│   │   │   ├── EnvironmentParams.tsx   # Temp, pressure, RH
│   │   │   ├── GasParams.tsx           # H2SO4, organics, NH3
│   │   │   ├── ProcessConfig.tsx       # Enable/disable processes + methods
│   │   │   └── PresetSelector.tsx      # Dropdown of preset scenarios
│   │   ├── Results/
│   │   │   ├── ResultsPanel.tsx        # Container for all result views
│   │   │   ├── SizeDistPlot.tsx        # Initial vs final dN/dlogDp
│   │   │   ├── BananaPlot.tsx          # Time-diameter heatmap
│   │   │   ├── TimeSeriesPlot.tsx      # N_tot, M_tot, Gc evolution
│   │   │   ├── ConservationPlot.tsx    # Mass/number conservation check
│   │   │   └── DownloadButton.tsx      # Export CSV/JSON
│   │   └── common/
│   │       ├── LoadingSpinner.tsx
│   │       ├── ErrorBanner.tsx
│   │       ├── Tooltip.tsx             # Parameter help text
│   │       └── RangeSlider.tsx         # Log-scale slider
│   ├── hooks/
│   │   ├── useSimulation.ts     # Submit + poll logic
│   │   └── usePresets.ts        # Fetch presets
│   ├── types/
│   │   └── simulation.ts        # TypeScript interfaces matching API
│   ├── utils/
│   │   ├── units.ts             # Display unit conversions
│   │   └── plotHelpers.ts       # Plotly layout/trace builders
│   └── constants/
│       └── parameterRanges.ts   # Min/max/default for all inputs
├── package.json
├── tsconfig.json
├── vite.config.ts               # Vite + React
├── .env.example                 # VITE_API_URL=https://api.tomas.example.com
├── Dockerfile
└── README.md
```

---

## 4. Backend API (tomas-api)

### 4.1 Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | **FastAPI** | Async, auto-docs (OpenAPI/Swagger), Pydantic validation |
| Runtime | **Uvicorn** | ASGI server, production-ready |
| Compute | **JAX (CPU)** | GPU optional; CPU is sufficient for single-scenario runs |
| Task queue | **In-memory** (start), **Redis + Celery** (scale) | Avoid infrastructure complexity initially |
| Caching | **In-memory LRU** | Cache compiled XLA programs + recent results |
| Container | **Docker** | Reproducible JAX environment |

### 4.2 API Endpoints

#### `POST /api/v1/simulate`

Submit a simulation job. Returns immediately with a job ID.

```json
// Request
{
  "aerosol": {
    "n_total_cm3": 1e5,
    "gmd_nm": 100,
    "gsd": 1.6
  },
  "environment": {
    "temp_K": 298,
    "pres_Pa": 101325,
    "rh": 0.5,
    "boxvol_cm3": 1e6
  },
  "gases": {
    "h2so4_molec_cm3": 1e7,
    "h2so4_prod_molec_cm3_s": 1e7,
    "org_molec_cm3": 1e7,
    "nh3_molec_cm3": 1e9,
    "fion_pairs_cm3_s": 3.0
  },
  "processes": {
    "nucleation": true,
    "coagulation": true,
    "condensation": true
  },
  "solver": {
    "cond_method": "ppm_jit",
    "coag_solver": "euler",
    "n_coag_substeps": 3,
    "dt_s": 60,
    "total_time_s": 86400
  },
  "nucleation_config": {
    "enable_organic": true,
    "enable_inorganic": true,
    "fn_scale": 1.0
  },
  "output": {
    "snapshot_interval_s": 3600,
    "include_history": true
  }
}
```

```json
// Response (202 Accepted)
{
  "job_id": "abc123",
  "status": "queued",
  "estimated_time_s": 5
}
```

#### `GET /api/v1/results/{job_id}`

Poll for results. Returns status or full results when complete.

```json
// Response (running)
{
  "job_id": "abc123",
  "status": "running",
  "progress": 0.45
}

// Response (complete)
{
  "job_id": "abc123",
  "status": "complete",
  "wall_time_s": 2.3,
  "results": {
    "time_s": [0, 3600, 7200, ...],
    "Nk": [[...], [...]],                     // (n_snapshots, nbins)
    "Mk_dry": [[...], [...]],                 // (n_snapshots, nbins) — total dry mass
    "N_tot": [1e11, 9.8e10, ...],             // Total number per snapshot
    "M_dry_tot": [1.2e-9, 1.3e-9, ...],      // Total dry mass per snapshot
    "Gc_h2so4": [1e-14, 9e-15, ...],         // H2SO4 gas per snapshot
    "bin_diameters_nm": [3.0, 3.78, ...],     // Geometric mean of bin edges
    "bin_edges_nm": [2.68, 3.38, ...],        // Bin boundary diameters
    "history": {                               // Per-minute (if requested)
      "time_s": [0, 60, 120, ...],
      "N_tot": [...],
      "M_dry_tot": [...],
      "Gc_h2so4": [...]
    },
    "diagnostics": {
      "mass_conservation_error": 1.2e-15,
      "number_conservation_error": 3.4e-7,
      "solver_info": "ppm_jit + euler coag"
    }
  }
}
```

#### `GET /api/v1/presets`

Return curated preset scenarios.

```json
{
  "presets": [
    {
      "id": "clean_background",
      "name": "Clean Background",
      "description": "Low aerosol, moderate H2SO4 — typical remote continental",
      "params": { ... }
    },
    {
      "id": "urban_polluted",
      "name": "Urban Polluted",
      "description": "High aerosol loading, strong condensation sink",
      "params": { ... }
    },
    {
      "id": "nucleation_event",
      "name": "New Particle Formation Event",
      "description": "Low CS + high H2SO4 → banana plot growth",
      "params": { ... }
    },
    ...
  ]
}
```

Suggested presets (derived from the 50-scenario benchmark):

| Preset | N_total | GMD | H2SO4 | Key Feature |
|--------|---------|-----|-------|-------------|
| Clean background | 1e3 /cm³ | 80 nm | 1e6 molec/cm³ | Low CS, slow growth |
| Urban polluted | 1e5 /cm³ | 200 nm | 5e7 molec/cm³ | High CS, fast condensation |
| NPF event | 1e4 /cm³ | 20 nm | 1e8 molec/cm³ | Strong nucleation banana |
| Coagulation-dominated | 1e6 /cm³ | 50 nm | 1e5 molec/cm³ | Rapid N loss, GSD broadening |
| Free troposphere | 1e2 /cm³ | 60 nm | 5e6 molec/cm³ | Low T (240K), low P (30 kPa) |
| Ternary nucleation | 1e3 /cm³ | 80 nm | 5e7 molec/cm³ | High NH3 (1e10), ternary path |

#### `GET /api/v1/health`

```json
{
  "status": "healthy",
  "jit_warmed_up": true,
  "uptime_s": 3600,
  "active_jobs": 2,
  "version": "0.1.0"
}
```

### 4.3 Request Validation

Physical plausibility checks (return 422 with clear error messages):

```python
# Hard limits — reject if violated
assert 1e0 <= n_total_cm3 <= 1e8       # Physically unreasonable otherwise
assert 1 <= gmd_nm <= 10000             # 1 nm to 10 µm
assert 1.05 <= gsd <= 4.0              # Near-monodisperse to very broad
assert 180 <= temp_K <= 350             # Stratosphere to hot surface
assert 1000 <= pres_Pa <= 110000       # ~30 km altitude to sea level
assert 0.0 <= rh <= 1.0                # Relative humidity fraction
assert 0 <= h2so4_molec_cm3 <= 1e10    # Up to extreme pollution
assert 0 <= h2so4_prod <= 1e10         # Production rate
assert 0 <= org_molec_cm3 <= 1e10      # Organic vapor
assert 0 <= nh3_molec_cm3 <= 1e12      # NH3 — highly variable
assert 0 <= fion <= 50                  # Ion-pair production rate
assert 1 <= dt_s <= 300                 # Time step
assert 60 <= total_time_s <= 172800     # 1 minute to 48 hours

# Soft limits — warn but allow
if n_total_cm3 > 1e7:
    warn("Extremely high aerosol loading — simulation may be slow")
if total_time_s > 86400:
    warn("Simulations >24h are resource-intensive")
```

### 4.4 JIT Warmup Strategy

JAX compiles functions on first call. Without warmup, the first user request would take 30–60 seconds.

```python
# app/services/warmup.py
async def warmup_jit():
    """Run during FastAPI lifespan startup."""
    # 1. Warm PPM condensation scan (most common path)
    run_condensation_scan(dummy_Nk, dummy_Mk, dummy_Gc, ..., nsteps=10, ...)

    # 2. Warm Euler coagulation
    coag_euler_step(dummy_Nk, dummy_Mk, ...)

    # 3. Warm full scan (nucl + coag + cond)
    run_full_scan(dummy_Nk, dummy_Mk, dummy_Gc, ..., nsteps=10, ...)

    # 4. Warm nucleation step
    nucleation_step(dummy_Nk, dummy_Mk, dummy_Gc, ...)

    # Total warmup: ~30-60s on CPU
    # Must run at server start, not per-request
```

**Precaution:** Warmup must use the same array shapes as production (NBINS=40, ICOMP=44). Different shapes trigger recompilation.

### 4.5 Job Management

```python
# app/services/job_manager.py

class JobManager:
    """In-memory async job manager.

    For production scale, replace with Redis + Celery.
    """
    def __init__(self, max_concurrent=2, max_queue=20, result_ttl_s=3600):
        self.jobs: dict[str, Job] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def submit(self, params: SimulationRequest) -> str:
        job_id = uuid4().hex[:12]
        # Reject if queue is full
        if len(self.active_jobs) >= self.max_queue:
            raise HTTPException(429, "Server busy — try again later")
        asyncio.create_task(self._run(job_id, params))
        return job_id

    async def _run(self, job_id, params):
        async with self.semaphore:
            # Run simulation in thread pool (JAX releases GIL during XLA)
            result = await asyncio.to_thread(run_simulation, params)
            self.jobs[job_id].result = result
            self.jobs[job_id].status = "complete"
```

### 4.6 `tomas-jax` as a Dependency

The API repo should install `tomas-jax` as a pip package:

```
# requirements.txt
tomas-jax @ git+https://github.com/yourorg/tomas-jax-coagulation.git
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.0
```

Or, for development, pip install the local checkout:

```bash
pip install -e /path/to/tomas-jax-coagulation
```

**Precaution:** Pin the `tomas-jax` version/commit hash in production to avoid breaking changes.

---

## 5. Frontend Application (tomas-web)

### 5.1 Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | **React 18+** with TypeScript | Type safety, ecosystem, hiring pool |
| Build tool | **Vite** | Fast dev server, optimized production builds |
| Charts | **Plotly.js** (react-plotly.js) | Scientific plotting, log axes, heatmaps, hover data |
| Styling | **Tailwind CSS** | Rapid prototyping, consistent design |
| State | **React hooks** (useState/useReducer) | Simple enough — no Redux needed |
| HTTP | **fetch** + custom wrapper | No need for axios at this scale |

### 5.2 Page Layout

```
┌──────────────────────────────────────────────────────────┐
│  TOMAS-JAX Box Model Simulator          [Docs] [GitHub]  │
├──────────────────────┬───────────────────────────────────┤
│                      │                                   │
│  ┌────────────────┐  │  ┌─────────────────────────────┐  │
│  │ Preset: [▼]    │  │  │                             │  │
│  ├────────────────┤  │  │   Size Distribution Plot    │  │
│  │ Aerosol        │  │  │   (dN/dlogDp vs Dp)        │  │
│  │  N_total ____  │  │  │   Initial + Final           │  │
│  │  GMD     ____  │  │  │                             │  │
│  │  GSD     ____  │  │  ├─────────────────────────────┤  │
│  ├────────────────┤  │  │                             │  │
│  │ Environment    │  │  │   Banana Plot               │  │
│  │  Temp    ____  │  │  │   (Time × Dp heatmap)      │  │
│  │  Pres    ____  │  │  │                             │  │
│  │  RH      ____  │  │  ├─────────────────────────────┤  │
│  ├────────────────┤  │  │                             │  │
│  │ Gas Phase      │  │  │   Time Series               │  │
│  │  H2SO4   ___   │  │  │   N_tot / M_tot / Gc       │  │
│  │  Prod    ___   │  │  │                             │  │
│  │  Org     ___   │  │  ├─────────────────────────────┤  │
│  │  NH3     ___   │  │  │   Conservation Diagnostic   │  │
│  │  Ions    ___   │  │  │   ΔM/M₀ and ΔN/N₀ vs time  │  │
│  ├────────────────┤  │  │                             │  │
│  │ Processes      │  │  └─────────────────────────────┘  │
│  │  [✓] Nucleation│  │                                   │
│  │  [✓] Coagulat. │  │  ┌─────────────────────────────┐  │
│  │  [✓] Condensat.│  │  │ Diagnostics                 │  │
│  │  Method: [▼]   │  │  │  Wall time: 2.3s            │  │
│  ├────────────────┤  │  │  Mass err: 1.2e-15          │  │
│  │                │  │  │  Solver: PPM + Euler         │  │
│  │ [Run Simulation│  │  │  [Download CSV] [Download NPZ│  │
│  │                │  │  └─────────────────────────────┘  │
│  └────────────────┘  │                                   │
│                      │                                   │
└──────────────────────┴───────────────────────────────────┘
```

### 5.3 Input Controls

For scientific parameters spanning orders of magnitude, use **log-scale sliders with numeric input override**:

| Parameter | Input Type | Display Unit | Internal Unit | Default |
|-----------|-----------|-------------|---------------|---------|
| N_total | Log slider + input | #/cm³ | #/cm³ | 1e5 |
| GMD | Log slider + input | nm | nm | 100 |
| GSD | Linear slider + input | — | — | 1.6 |
| Temperature | Linear slider + input | K | K | 298 |
| Pressure | Linear slider + input | hPa | Pa | 1013.25 |
| RH | Linear slider + input | % | fraction | 50 |
| H2SO4 | Log slider + input | molec/cm³ | molec/cm³ | 1e7 |
| H2SO4 prod | Log slider + input | molec/cm³/s | molec/cm³/s | 1e7 |
| Organic | Log slider + input | molec/cm³ | molec/cm³ | 1e7 |
| NH3 | Log slider + input | molec/cm³ | molec/cm³ | 1e9 |
| Ion rate | Linear slider + input | pairs/cm³/s | pairs/cm³/s | 3.0 |
| Total time | Dropdown | hours | s | 24 |
| Timestep | Dropdown | seconds | s | 60 |

**UX recommendations:**
- Show parameter tooltips explaining physical meaning ("GMD = Geometric Mean Diameter of the initial lognormal distribution")
- Gray out nucleation parameters when nucleation is disabled
- Show a "parameter summary" card that confirms the simulation before running
- Support URL query parameters for sharing configurations (e.g., `?preset=npf_event`)

### 5.4 Visualization Details

#### Size Distribution (dN/dlogDp)

```
X-axis: Particle diameter [nm], log scale, range 1–10000 nm
Y-axis: dN/dlogDp [#/cm³], log scale
Traces: Initial (dashed blue), Final (solid red)
```

Conversion from internal units:
```typescript
// Bin geometric mean diameters
const dp_nm = binEdgesNm.map((e, i) =>
  Math.sqrt(e * binEdgesNm[i + 1])  // geometric mean
);

// dN/dlogDp = Nk / (boxvol * dlogDp)
const dlogDp = Math.log10(binEdgesNm[1] / binEdgesNm[0]);  // constant for doubling grid
const dNdlogDp = Nk.map(n => n / (boxvol * dlogDp));
```

#### Banana Plot

```
X-axis: Time [hours]
Y-axis: Particle diameter [nm], log scale
Color: dN/dlogDp [#/cm³], log color scale
Type: Plotly heatmap or contour
```

This is the signature plot for nucleation events — new particles appearing at ~3 nm and growing to 50+ nm over 12–24 hours.

#### Time Series

```
Panel 1: N_tot [#/cm³] vs time [hours]
Panel 2: M_dry [µg/m³] vs time [hours]
Panel 3: H2SO4 gas [molec/cm³] vs time [hours]
```

#### Conservation Diagnostic

```
X-axis: Time [hours]
Y-axis: Relative error (dimensionless)
Traces: (M_dry(t) + Gc(t) - M_dry(0) - Gc(0)) / (M_dry(0) + Gc(0))
```

### 5.5 State Management

```typescript
interface SimulationState {
  // Input
  params: SimulationParams;
  preset: string | null;

  // Job tracking
  jobId: string | null;
  status: 'idle' | 'submitting' | 'queued' | 'running' | 'complete' | 'error';
  progress: number;        // 0-1

  // Output
  results: SimulationResults | null;
  error: string | null;
}
```

### 5.6 Polling Strategy

```typescript
// hooks/useSimulation.ts
async function pollResults(jobId: string) {
  const POLL_INTERVALS = [500, 500, 1000, 1000, 2000, 2000, 3000]; // ms
  let attempt = 0;

  while (true) {
    const res = await fetch(`${API_URL}/results/${jobId}`);
    const data = await res.json();

    if (data.status === 'complete') return data.results;
    if (data.status === 'error') throw new Error(data.error);

    const delay = POLL_INTERVALS[Math.min(attempt, POLL_INTERVALS.length - 1)];
    await sleep(delay);
    attempt++;

    if (attempt > 120) throw new Error('Simulation timed out');
  }
}
```

**Alternative: Server-Sent Events (SSE).** For a better UX, the backend can stream progress updates:

```
GET /api/v1/results/{job_id}/stream

data: {"status": "running", "progress": 0.3, "step": 432}
data: {"status": "running", "progress": 0.7, "step": 1008}
data: {"status": "complete", "results": {...}}
```

This avoids polling and gives real-time progress updates. FastAPI supports SSE natively via `StreamingResponse`.

---

## 6. Data Flow

### 6.1 Request → Response Lifecycle

```
User clicks "Run"
      │
      ▼
Frontend validates inputs (client-side)
      │
      ▼
POST /api/v1/simulate  ──────────────────────►  Backend
      │                                            │
      ▼                                            ▼
Receive job_id (202)                     Validate (Pydantic)
      │                                            │
      ▼                                            ▼
Start polling GET /results/{id}          Convert units (user → internal)
      │                                            │
      ▼                                            ▼
Show progress spinner                    Build TomasState.create(...)
      │                                            │
      ▼                                            ▼
Receive results (200)                    Run scan-fused simulation
      │                                      (0.3–13s depending on mode)
      ▼                                            │
Convert units (internal → display)               ▼
      │                                    Convert outputs (internal → user)
      ▼                                            │
Render Plotly charts                             ▼
                                          Return JSON results
```

### 6.2 Unit Conversion Layer

The API accepts scientist-friendly units and converts internally:

```python
# Backend: user units → internal units
Nk_internal = lognormal_Nk(n_total_cm3, gmd_m, gsd) * boxvol_cm3  # #/cm³ → #/cell
Gc_h2so4_kg = h2so4_molec_cm3 * boxvol_cm3 * (MW_H2SO4/1000) / AVOGADRO
prod_rate_kg = h2so4_prod_molec_cm3_s * boxvol_cm3 * (MW_H2SO4/1000) / AVOGADRO

# Backend: internal units → response units
Nk_cm3 = Nk_internal / boxvol_cm3         # #/cell → #/cm³
dp_nm = (6 * xk_kg / (pi * rho)) ** (1/3) * 1e9  # kg → nm
M_ug_m3 = M_kg_cell / boxvol_cm3 * 1e6 * 1e9      # kg/cell → µg/m³
```

**Precaution:** All unit conversions happen in the backend. The frontend only deals with display units (nm, #/cm³, µg/m³, molec/cm³). This prevents conversion bugs from creeping into the UI code.

### 6.3 Response Size Estimation

For a 24-hour simulation with hourly snapshots:

| Field | Shape | Size (float64) | Size (JSON) |
|-------|-------|----------------|-------------|
| Nk | (24, 36) | 6.9 KB | ~20 KB |
| Mk_dry | (24, 36) | 6.9 KB | ~20 KB |
| N_tot | (24,) | 192 B | ~600 B |
| M_dry_tot | (24,) | 192 B | ~600 B |
| Gc_h2so4 | (24,) | 192 B | ~600 B |
| history (1440 × 3) | (1440, 3) | 34.6 KB | ~100 KB |
| **Total** | | **~50 KB** | **~150 KB** |

Well within typical API limits. No need for binary transport unless we add per-minute full Nk snapshots (which would be 1440 × 36 = ~400 KB).

---

## 7. Deployment

### 7.1 Frontend Deployment

**Recommended: Vercel (free tier)**

```bash
# Deploy
cd tomas-web
npm run build          # Produces dist/
vercel deploy          # Or connect GitHub repo for auto-deploy
```

- Free tier: 100 GB bandwidth/month, unlimited deploys
- Automatic HTTPS, CDN, preview deployments per PR
- Environment variable: `VITE_API_URL=https://api.tomas.example.com`

**Alternatives:** Netlify (equivalent), Cloudflare Pages (faster global CDN), GitHub Pages (simple but no server-side features).

### 7.2 Backend Deployment

**Option A: Fly.io (recommended for simplicity)**

```toml
# fly.toml
[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8000
  force_https = true

[[vm]]
  cpu_kind = "shared"
  cpus = 2
  memory_mb = 2048       # JAX needs memory for XLA compilation
```

- ~$7/month for 2-CPU shared VM
- Auto-sleep (scale to zero when idle) — saves cost
- But: cold starts re-trigger JIT warmup (~60s)

**Option B: GCP Cloud Run**

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/tomas-api', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/tomas-api']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args: ['gcloud', 'run', 'deploy', 'tomas-api',
           '--image', 'gcr.io/$PROJECT_ID/tomas-api',
           '--memory', '2Gi', '--cpu', '2',
           '--min-instances', '1',     # Avoid cold starts
           '--max-instances', '4',     # Auto-scale
           '--timeout', '120']
```

- `min-instances=1` keeps one warm instance (avoids JIT cold start)
- Auto-scales to 4 instances under load
- ~$15–30/month with min-instance always on

**Option C: Small VM (EC2/GCE)**

- t3.medium (2 vCPU, 4 GB) — ~$30/month
- Always warm, simplest to manage
- No auto-scaling; add a load balancer later if needed

### 7.3 Dockerfile

```dockerfile
# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ app/

# JAX CPU-only (no GPU libs needed)
ENV JAX_PLATFORM_NAME=cpu
ENV JAX_ENABLE_X64=True

# Warmup on startup (in lifespan)
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.4 Domain & CORS

```python
# app/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tomas.example.com",       # Production frontend
        "http://localhost:5173",            # Vite dev server
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

**Custom domain setup:**
- Frontend: `tomas.example.com` → Vercel
- Backend: `api.tomas.example.com` → Fly.io / Cloud Run
- Use the same parent domain to simplify CORS

---

## 8. Security

### 8.1 Rate Limiting

```python
# Option 1: In-app (simple)
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/simulate")
@limiter.limit("10/minute")       # Per IP
@limiter.limit("100/hour")        # Per IP
async def simulate(request: SimulationRequest):
    ...

# Option 2: Reverse proxy (production)
# Cloudflare / nginx rate limiting at the edge
```

### 8.2 Input Sanitization

- **No arbitrary code execution** — the API only accepts numeric parameters through Pydantic models
- **Bounded parameter ranges** — all inputs validated against physical limits (see §4.3)
- **Maximum simulation time** — cap at 48 hours (172800 seconds) to prevent resource exhaustion
- **Maximum concurrent jobs** — limit to 2–4 per server instance
- **Job timeout** — kill simulations that exceed 120 seconds (full mode with Tsit5 can be slow)

### 8.3 Denial of Service Considerations

Worst-case simulation parameters (adversarial inputs designed to maximize compute):
- N_total = 1e8, full mode (nucl+coag+cond), total_time = 48h → could run 30+ seconds
- Mitigation: enforce `total_time_s <= 86400` for full mode, use Euler coag (not Tsit5)

```python
# Enforce solver constraints
if "coagulation" in processes and total_time_s > 86400:
    raise HTTPException(400,
        "Simulations with coagulation are limited to 24 hours. "
        "Use condensation-only for longer runs.")

if coag_solver == "tsit5":
    raise HTTPException(400,
        "Tsit5 solver is not available via the web interface (too slow). "
        "Use euler solver instead.")
```

### 8.4 No Authentication (Phase 1)

For a public research tool, start without authentication. Add API keys later if abuse becomes a problem.

If needed later:
- GitHub OAuth for user accounts (researchers already have GitHub)
- API key for programmatic access
- Usage quotas per user

---

## 9. Performance & Scalability

### 9.1 Expected Latencies

| Mode | Processes | Wall Time | Notes |
|------|-----------|-----------|-------|
| Condensation only (PPM) | cond | ~0.3s | Fastest path |
| Condensation only (TFL) | cond | ~0.5s | Slightly slower |
| Coagulation + Condensation | coag+cond | ~0.5s | Euler + PPM |
| Nucl + Condensation | nucl+cond | ~0.5s | Scan-fused |
| Full (nucl+coag+cond) | all | ~5–13s | Most expensive |

Add ~200ms for HTTP overhead, JSON serialization, and unit conversion.

### 9.2 Concurrent Users

With a 2-CPU instance running 2 concurrent simulations:

| Scenario | Users/minute | Notes |
|----------|-------------|-------|
| All cond-only | ~200 | 0.3s each, 2 concurrent |
| All full mode | ~10 | 10s each, 2 concurrent |
| Mixed (80% cond, 20% full) | ~30 | Weighted average |

For a research tool, 10–30 users/minute is more than sufficient. Most academic tools see <100 users/day.

### 9.3 Scaling Strategy

**Phase 1 (launch):** Single 2-CPU instance, 2 concurrent jobs, in-memory queue.

**Phase 2 (if needed):**
- Add Redis for job queue persistence
- Auto-scale to 2–4 backend instances behind a load balancer
- Add result caching — hash the request parameters, serve cached results for identical inputs

**Phase 3 (unlikely needed):**
- GPU instance for JAX (10x speedup over CPU for large batch)
- Precomputed result database for preset scenarios

### 9.4 Cold Start Mitigation

JAX JIT compilation takes 30–60 seconds on first call. Strategies:

1. **Warmup in lifespan** (required): Run dummy simulations at server start
2. **Keep-alive** (recommended): Ping `/health` every 5 minutes to prevent auto-sleep
3. **Min instances = 1** (if using Cloud Run): Always keep one warm container
4. **AOT compilation** (advanced): Use `jax.jit(fn).lower(args).compile()` to cache XLA programs, or serialize compiled functions to disk

---

## 10. Cost Estimation

### Monthly Costs (Phase 1)

| Component | Service | Cost |
|-----------|---------|------|
| Frontend hosting | Vercel free tier | $0 |
| Backend compute | Fly.io shared-2x (2 CPU, 2 GB) | ~$7–15 |
| Domain | `.com` or `.org` | ~$1 |
| SSL | Included (Let's Encrypt) | $0 |
| **Total** | | **~$8–16/month** |

### If scaling is needed

| Component | Service | Cost |
|-----------|---------|------|
| Frontend | Vercel Pro | $20/month |
| Backend | Cloud Run (2 CPU, 2 GB, min=1, max=4) | ~$30–60 |
| Redis (job queue) | Upstash free tier | $0 |
| Monitoring | Sentry free tier | $0 |
| **Total** | | **~$50–80/month** |

---

## 11. Development Roadmap

### Phase 1: MVP (2–3 weeks)

**Backend:**
- [ ] FastAPI project scaffolding with health endpoint
- [ ] Pydantic models for request/response
- [ ] Simulation service wrapping `run_box_model` logic
- [ ] JIT warmup on startup
- [ ] In-memory job queue (submit + poll)
- [ ] Unit conversion layer
- [ ] 6 preset scenarios
- [ ] Dockerfile + deploy to Fly.io
- [ ] CORS configuration

**Frontend:**
- [ ] Vite + React + TypeScript scaffolding
- [ ] Simulation form with all parameters
- [ ] Preset selector dropdown
- [ ] Submit + poll + loading state
- [ ] Size distribution plot (Plotly)
- [ ] Time series plot (N_tot, M_tot, Gc)
- [ ] Diagnostics panel (wall time, conservation error)
- [ ] Deploy to Vercel

**Deliverable:** Working public URL where researchers can run condensation-only simulations and see results.

### Phase 2: Full Features (2 weeks)

- [ ] Banana plot visualization
- [ ] Conservation diagnostic plot
- [ ] All process modes (nucl, coag, cond, combined, full)
- [ ] Method selection (PPM vs TFL)
- [ ] CSV/JSON download of results
- [ ] SSE for real-time progress
- [ ] URL-based parameter sharing
- [ ] Mobile-responsive layout

### Phase 3: Polish (1–2 weeks)

- [ ] Parameter tooltips with physical explanations
- [ ] "About" page with model description and references
- [ ] Error handling for edge cases (server down, timeout, invalid results)
- [ ] Rate limiting
- [ ] Basic analytics (how many simulations run per day)
- [ ] OpenAPI docs page (`/docs`)

### Phase 4: Advanced (future)

- [ ] Multi-scenario comparison (run 2 simulations, overlay results)
- [ ] Custom bin grid (40/80/160 bins)
- [ ] Batch mode (sweep one parameter)
- [ ] User accounts + saved simulations
- [ ] GPU backend for faster full-mode simulations
- [ ] Jupyter notebook export

---

## 12. Risks & Mitigations

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| JAX cold start (60s) | Poor first-request UX | High | Warmup in lifespan; keep-alive pings; min-instances=1 |
| Full-mode simulation timeout | User thinks it's broken | Medium | Show progress bar; set 120s timeout with clear message; default to fast modes |
| JAX version mismatch | Subtle numerical differences | Low | Pin JAX version in requirements.txt; test in Docker locally |
| Memory leak from JIT cache | Server OOM after many unique shapes | Low | Only support NBINS=40/80; monitor memory; restart weekly |
| Plotly.js bundle size (3.5 MB) | Slow frontend load | Medium | Use partial bundle (only scatter + heatmap); lazy-load |
| Concurrent JAX on single CPU | Simulations interfere | Medium | Semaphore limits concurrency to 2; thread pool for isolation |

### Operational Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Abuse (crypto mining, DoS) | Server overload, high costs | Low | Rate limiting; job timeout; no arbitrary code paths |
| Upstream tomas-jax breaking change | API returns wrong results | Medium | Pin commit hash; integration tests in API CI |
| Server goes down unnoticed | Users see errors | Medium | Health check endpoint; uptime monitor (UptimeRobot, free) |
| Cost overrun | Unexpected bill | Low | Set billing alerts; Fly.io has spending caps |

### Scientific Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Users misinterpret results | Wrong conclusions in papers | Medium | Add disclaimers; explain limitations; show conservation diagnostics |
| Parameter combos that crash solver | 500 error | Medium | Exhaustive input validation; catch JAX errors; return meaningful messages |
| Numerical differences from Fortran | Users question correctness | Low | Document known discrepancies (organic clamping, etc.); link to validation paper |

---

## Appendix A: API Reference

### Full OpenAPI Spec (auto-generated by FastAPI)

Available at `https://api.tomas.example.com/docs` (Swagger UI) and `https://api.tomas.example.com/redoc` (ReDoc).

### Error Codes

| HTTP Code | Meaning | Example |
|-----------|---------|---------|
| 200 | Success | Results ready |
| 202 | Accepted | Job submitted, not yet complete |
| 400 | Bad Request | Invalid parameter combination |
| 404 | Not Found | Unknown job_id |
| 422 | Validation Error | Parameter out of range |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Error | JAX crash (unexpected) |
| 503 | Service Unavailable | Server warming up or overloaded |

---

## Appendix B: Parameter Ranges

### Input Parameters

| Parameter | Min | Max | Scale | Default | Unit | Physical Context |
|-----------|-----|-----|-------|---------|------|-----------------|
| n_total | 1 | 1e8 | log | 1e5 | #/cm³ | Clean marine ~300, urban ~10⁴–10⁵ |
| gmd | 1 | 10000 | log | 100 | nm | Nucleation mode ~3–10, accumulation ~100–300 |
| gsd | 1.05 | 4.0 | linear | 1.6 | — | Monodisperse ~1.1, typical ~1.5–1.8 |
| temp | 180 | 350 | linear | 298 | K | Stratosphere ~200, tropical surface ~310 |
| pres | 1000 | 110000 | linear | 101325 | Pa | ~30 km to sea level |
| rh | 0 | 100 | linear | 50 | % | Desert ~10%, tropical ~90% |
| h2so4 | 0 | 1e10 | log | 1e7 | molec/cm³ | Clean ~10⁵, polluted ~10⁸ |
| h2so4_prod | 0 | 1e10 | log | 1e7 | molec/cm³/s | SO₂ photochemistry rate |
| org | 0 | 1e10 | log | 1e7 | molec/cm³ | Biogenic ~10⁶, urban ~10⁸ |
| nh3 | 0 | 1e12 | log | 1e9 | molec/cm³ | Remote ~10⁸, agricultural ~10¹¹ |
| fion | 0 | 50 | linear | 3.0 | pairs/cm³/s | Ground ~2–5, high altitude ~30 |
| dt | 1 | 300 | discrete | 60 | s | Numerical stability vs speed |
| total_time | 60 | 172800 | discrete | 86400 | s | 1 min to 48 hours |

### Process Modes (Derived from Input)

| Mode | Processes | Typical Use Case |
|------|-----------|-----------------|
| `cond_only` | Condensation | Studying condensation growth |
| `coag_only` | Coagulation | Studying coagulation kernel |
| `combined` | Coag + Cond | Standard aging simulation |
| `nucl_cond` | Nucl + Cond | New particle formation event |
| `full` | Nucl + Coag + Cond | Complete box model |

---

## Appendix C: References

1. Adams, P.J. and Seinfeld, J.H. (2002). Predicting global aerosol size distributions in general circulation models. *J. Geophys. Res.*, 107(D19).
2. Riccobono, F. et al. (2014). Oxidation products of biogenic emissions contribute to nucleation of atmospheric particles. *Science*, 344(6185), 717–721.
3. Dunne, E.M. et al. (2016). Global atmospheric particle formation from CERN CLOUD measurements. *Science*, 354(6316), 1119–1124.
4. Colella, P. and Woodward, P.R. (1984). The Piecewise Parabolic Method (PPM) for gas-dynamical simulations. *J. Comput. Phys.*, 54(1), 174–201.
