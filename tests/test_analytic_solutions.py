"""Tests against exactly solvable limits of the transmon-cavity model
(sanity checks 1-3 of notebooks/sanity_checks.ipynb).

With U = 0 the model is linear in the cavity (a) and transmon (b) operators, so
coherent states stay coherent for every noise realization, with amplitudes
    d<a>/dt = epsilon - k <a> - Omega <b>,    d<b>/dt = Omega <a>.
With k = epsilon = U = 0 there is no noise and transmon and cavity exchange
excitations without loss."""

import numpy as np
import pytest

from openquantum_sde.integrators import splittingExactIterativeCN
from openquantum_sde.simulation import simulate_fixed_dt
from openquantum_sde.systems import TransmonCavity
from openquantum_sde.utils import calculate_num_photons


# Sanity check 1: driven damped cavity (single transmon level, Omega = U = 0)
CAVITY = dict(M=1, N=51, k=1.0, Omega=0.0, epsilon=5.0, U=0.0)
# Sanity check 2: driven damped cavity coupled to an undamped transmon (U = 0)
COUPLED = dict(M=8, N=8, k=1.0, Omega=10.0, epsilon=4.0, U=0.0)
# Sanity check 3: lossless exchange between transmon and cavity (k = epsilon = U = 0)
LOSSLESS = dict(M=3, N=3, k=0.0, Omega=1.0, epsilon=0.0, U=0.0)


def simulate(system, make_integrator, T, dt, initial=(0, 0), calculate_current=False, seed=0):
    '''Fixed-dt run from basis state |m, n> = initial, saving every 0.1 time units'''
    X0 = np.zeros((system.M, system.N), dtype=np.complex128)
    X0[initial] = 1.0
    return simulate_fixed_dt(X0, round(T / dt), dt, calculate_current=calculate_current,
                             save_every=round(0.1 / dt), renormalize_every=1, progress_bar=False,
                             seed=seed, integrator=make_integrator(system.M, system.N), system=system)


def late_time_mean_current(system, t_settle=10.0, T=210.0, dt=2e-3):
    '''Time average of the current alpha after it settles (main integrator, seeded)'''
    _, t, _, current = simulate(system, lambda M, N: splittingExactIterativeCN(), T, dt,
                                calculate_current=True)
    return current[t > t_settle].mean()


def cavity_amplitude(X):
    '''<a> = sum_mn conj(X[m, n]) sqrt(n+1) X[m, n+1]'''
    return np.sum(np.conj(X[:, :-1]) * np.sqrt(np.arange(1, X.shape[1])) * X[:, 1:])


def transmon_amplitude(X):
    '''<b> = sum_mn conj(X[m, n]) sqrt(m+1) X[m+1, n]'''
    return np.sum(np.conj(X[:-1, :]) * np.sqrt(np.arange(1, X.shape[0]))[:, None] * X[1:, :])


def coupled_amplitudes_exact(t, system):
    '''Solution of the amplitude equations from vacuum: returns (<a>(t), <b>(t)).
    Steady state: <a> = 0 and <b> = epsilon/Omega'''
    A = np.array([[-system.k, -system.Omega], [system.Omega, 0.0]])
    steady = np.array([0.0, system.epsilon / system.Omega])
    w, V = np.linalg.eig(A)
    c = np.linalg.solve(V, -steady)
    amplitudes = np.array([(V @ (np.exp(w * ti) * c)).real for ti in t]) + steady
    return amplitudes[:, 0], amplitudes[:, 1]


# =========================
# Sanity check 1: driven damped cavity
# =========================

def test_driven_damped_cavity_photon_number_follows_analytic_curve(make_integrator):
    '''sqrt(<n>) = (epsilon/k)(1 - exp(-kt)), within 10% of epsilon/k'''
    system = TransmonCavity(**CAVITY)
    _, t, traj, _ = simulate(system, make_integrator, T=6.0, dt=5e-4)
    sqrt_n = np.sqrt([calculate_num_photons(X).real for X in traj])
    expected = system.epsilon / system.k * (1 - np.exp(-system.k * t))
    assert np.max(np.abs(sqrt_n - expected)) < 0.1 * system.epsilon / system.k


def test_driven_damped_cavity_current_averages_to_epsilon_over_k():
    system = TransmonCavity(**CAVITY)
    assert late_time_mean_current(system) == pytest.approx(system.epsilon / system.k, abs=0.2)


# =========================
# Sanity check 2: cavity coupled to transmon
# =========================

def test_coupled_oscillators_amplitudes_follow_analytic_solution(make_integrator):
    system = TransmonCavity(**COUPLED)
    _, t, traj, _ = simulate(system, make_integrator, T=12.0, dt=5e-4)
    a_exact, b_exact = coupled_amplitudes_exact(t, system)
    assert np.max(np.abs([cavity_amplitude(X) for X in traj] - a_exact)) < 0.03
    assert np.max(np.abs([transmon_amplitude(X) for X in traj] - b_exact)) < 0.03


def test_coupled_oscillators_current_averages_to_zero():
    '''The cavity empties (<a> -> 0), so the current averages to zero'''
    assert late_time_mean_current(TransmonCavity(**COUPLED)) == pytest.approx(0.0, abs=0.2)


# =========================
# Sanity check 3: lossless exchange
# =========================

def test_lossless_exchange_oscillates_without_decay(make_integrator):
    '''From |m=2, n=0>, with theta = Omega*t:
    X[2,0] = cos^2(theta), X[1,1] = -sqrt(2) cos(theta) sin(theta), X[0,2] = sin^2(theta),
    and all other states stay empty'''
    system = TransmonCavity(**LOSSLESS)
    _, t, traj, _ = simulate(system, make_integrator, T=20.0, dt=1e-3, initial=(2, 0))
    theta = system.Omega * t
    exact = np.zeros_like(traj)
    exact[:, 2, 0] = np.cos(theta)**2
    exact[:, 1, 1] = -np.sqrt(2) * np.cos(theta) * np.sin(theta)
    exact[:, 0, 2] = np.sin(theta)**2
    assert np.max(np.abs(traj - exact)) < 0.1


def test_current_is_zero_without_dissipation():
    '''With k = 0 no light leaves the cavity, so there is no measured current'''
    system = TransmonCavity(**LOSSLESS)
    _, _, _, current = simulate(system, lambda M, N: splittingExactIterativeCN(), T=1.0, dt=1e-3,
                                initial=(2, 0), calculate_current=True)
    assert np.all(current == 0)
