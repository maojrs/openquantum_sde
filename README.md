# openquantum_sde

Numerical integrators for stochastic Schrödinger equations (SSEs) in open quantum systems.

## Overview

- Simulates quantum trajectories of open quantum systems under continuous (heterodyne) measurement.
- The included system is a circuit-QED model (superconducting transmon coupled to a microwave cavity); the integrators are general.
- The state is a matrix `X[m, n]` of probability amplitudes (e.g. transmon level `m`, photon number `n`).
- Time stepping with fixed or adaptive `dt`, optional seeding for reproducible runs.
- Computes the filtered measurement current and observables (photon number, transmon populations).
- Kernels are compiled with [numba](https://numba.pydata.org/) for speed.

Package layout (`src/openquantum_sde/`):

| Module | Contents |
|---|---|
| `systems/` | Physical systems (drift and noise terms), e.g. `TransmonCavity` (circuit QED) |
| `integrators/` | SSE integrators (Euler–Maruyama, Milstein, Heun, splitting methods, RK4) |
| `simulation.py` | `simulate_fixed_dt`, `simulate_adaptive_dt` |
| `utils.py` | Noise, seeding, norms and observables |
| `io.py` | Save/load trajectories (HDF5) and parameters (JSON) |
| `plotting.py` | Plots of the current and transmon populations |

## Installation

Requires [uv](https://docs.astral.sh/uv/) (Python ≥ 3.12; the repository pins 3.14).

```bash
git clone https://github.com/maojrs/openquantum_sde.git
cd openquantum_sde
uv sync                      # creates .venv with the package and its dependencies
uv sync --group notebooks    # optional: also installs Jupyter for the notebooks
```

Run scripts inside the environment with `uv run`, e.g. `uv run python scripts/transmon_cavity_splitting.py`.

## Usage

### Fixed time step

```python
import numpy as np
from openquantum_sde.systems import TransmonCavity
from openquantum_sde.integrators import splittingExactIterativeCN
from openquantum_sde.simulation import simulate_fixed_dt
from openquantum_sde.utils import calculate_num_photons

# System: M transmon levels, N photon levels
M, N = 9, 101
system = TransmonCavity(M, N, k=1.0, Omega=50.0, epsilon=10.0, U=400.0)

# Initial state: transmon in its ground state, cavity in vacuum
X0 = np.zeros((M, N), dtype=np.complex128)
X0[0, 0] = 1.0

dt_array, times, traj, current = simulate_fixed_dt(
    X0, nsteps=20000, dt=2e-4,
    save_every=100, renormalize_every=1,
    calculate_current=True, seed=1,
    integrator=splittingExactIterativeCN(),
    system=system,
)

num_photons = [calculate_num_photons(X).real for X in traj]
```

- `traj[i]` is the state at `times[i]`; `current[i]` is the filtered measurement current.
- `seed` makes the run reproducible (omit it for a random run).

### Adaptive time step

```python
from openquantum_sde.simulation import simulate_adaptive_dt

dt_array, times, traj, current = simulate_adaptive_dt(
    X0, nsteps_max=20000, dt_min=5e-5, dt_max=2e-4, tol=0.8,
    save_every=100, renormalize_every=100,
    calculate_current=True, seed=1,
    integrator=splittingExactIterativeCN(),
    system=system,
)
```

### Saving and loading

```python
from pathlib import Path
from openquantum_sde.io import save_trajectory, load_trajectory

save_trajectory("traj_0001", Path("data"), times, traj, current, simid="0001")
times, traj, current = load_trajectory("traj_0001", Path("data"))
```

### Choosing an integrator

- **Recommended: `splittingExactIterativeCN`**: exact solution for the diagonal part, Crank–Nicolson for the rest; second order and robust.
- Also second order or higher: `splittingExactHeun`, `splittingExactMidpointEuler`, `stochasticHeun`, `splittingRK4EM`, `splittingRK4Milstein`.
- First order: `EulerMaruyama`, `Milstein`, `splittingExactEuler`, `splittingExactMilstein`. These can blow up for weakly damped, strongly coupled systems unless `dt` is very small.
- The `splittingExact*` integrators accept `taming=True` to tame the drift.
- `splittingRK4EM` and `splittingRK4Milstein` take the system size: `splittingRK4EM(M, N)`.

More examples are in `scripts/` and `notebooks/`.

## Tests

```bash
uv run pytest        # add -v to list every test
```

| File | Checks |
|---|---|
| `test_analytic_solutions.py` | Simulations match exactly solvable limits: driven damped cavity, cavity coupled to an undamped transmon, lossless transmon–cavity exchange (sanity checks of `notebooks/sanity_checks.ipynb`) |
| `test_integrators.py` | Every integrator converges with its expected order (1, 2 or 4); taming |
| `test_transmon_cavity.py` | Drift and noise routines agree with each other and conserve the norm without dissipation |
| `test_simulation.py` | Fixed and adaptive runs for all integrators: normalized states, `dt` within bounds, reproducible seeding |
| `test_utils.py` | Noise statistics, seeding, norms and observables |
| `test_io.py` | Save/load round trips |
| `test_package_exports.py` | Everything listed in `__all__` is importable |

## License

MIT, see [LICENSE](LICENSE).
