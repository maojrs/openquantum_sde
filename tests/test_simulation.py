"""Tests for the simulation drivers simulate_fixed_dt and simulate_adaptive_dt."""

import numpy as np
import pytest

from openquantum_sde.integrators import EulerMaruyama
from openquantum_sde.simulation import simulate_fixed_dt, simulate_adaptive_dt


NSTEPS, DT, SAVE_EVERY = 200, 1e-3, 10
DT_MIN, DT_MAX, TOL = 1e-5, 1e-3, 1e-3


def euler_maruyama(M, N):
    return EulerMaruyama()


def run_fixed(system, X0, make_integrator=euler_maruyama, seed=0):
    return simulate_fixed_dt(X0, NSTEPS, DT, calculate_current=True, save_every=SAVE_EVERY,
                             renormalize_every=1, progress_bar=False, seed=seed,
                             integrator=make_integrator(system.M, system.N), system=system)


def run_adaptive(system, X0, make_integrator=euler_maruyama, seed=0):
    return simulate_adaptive_dt(X0, NSTEPS, DT_MIN, DT_MAX, TOL, calculate_current=True,
                                save_every=1, renormalize_every=1, progress_bar=False, seed=seed,
                                integrator=make_integrator(system.M, system.N), system=system)


# =========================
# Every integrator
# =========================

def test_fixed_dt_returns_normalized_states_on_regular_times(make_integrator, system, vacuum):
    dt_array, times, traj, current = run_fixed(system, vacuum, make_integrator)
    nsave = NSTEPS // SAVE_EVERY + 1
    assert traj.shape == (nsave, system.M, system.N)
    assert times.shape == current.shape == (nsave,)
    assert np.all(np.isfinite(traj)) and np.all(np.isfinite(current))
    np.testing.assert_allclose(np.linalg.norm(traj, axis=(1, 2)), 1.0, rtol=1e-12)
    np.testing.assert_allclose(dt_array, DT, rtol=1e-9)


def test_adaptive_dt_varies_within_bounds(make_integrator, system, vacuum):
    dt_array, _, traj, _ = run_adaptive(system, vacuum, make_integrator)
    assert np.all(np.isfinite(traj))
    assert np.all((dt_array >= DT_MIN * (1 - 1e-9)) & (dt_array <= DT_MAX * (1 + 1e-9)))
    assert len(np.unique(np.round(dt_array, 12))) > 2


# =========================
# Seeding and input handling
# =========================

@pytest.mark.parametrize("run", [run_fixed, run_adaptive], ids=["fixed_dt", "adaptive_dt"])
def test_same_seed_gives_identical_runs(run, system, vacuum):
    first, repeat, other = run(system, vacuum, seed=1), run(system, vacuum, seed=1), run(system, vacuum, seed=2)
    for a, b in zip(first, repeat):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[2], other[2])


def test_real_initial_state_gives_same_result_as_complex(system, vacuum):
    from_complex = run_fixed(system, vacuum)
    from_real = run_fixed(system, vacuum.real.copy())
    for a, b in zip(from_complex, from_real):
        np.testing.assert_array_equal(a, b)
