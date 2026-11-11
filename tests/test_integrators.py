"""Tests for the integrators.

Without dissipation (k = 0) the noise vanishes and the equation reduces to the
linear Schrödinger equation dX/dt = A X, with exact solution X(T) = exp(A T) X0.
Each integrator must converge to it with its expected order."""

import numpy as np
import pytest

from openquantum_sde.integrators import (
    EulerMaruyama, Milstein, stochasticHeun,
    splittingExactEuler, splittingExactMidpointEuler, splittingExactIterativeCN,
    splittingExactHeun, splittingExactMilstein, splittingRK4EM, splittingRK4Milstein,
)
from openquantum_sde.systems import TransmonCavity


M, N, T = 3, 4, 0.4

# Integrator factory and expected order of convergence (deterministic part)
INTEGRATORS = {
    "EulerMaruyama":               (lambda: EulerMaruyama(), 1),
    "Milstein":                    (lambda: Milstein(), 1),
    "stochasticHeun":              (lambda: stochasticHeun(), 2),
    "splittingExactEuler":         (lambda: splittingExactEuler(), 1),
    "splittingExactMidpointEuler": (lambda: splittingExactMidpointEuler(), 2),
    "splittingExactIterativeCN":   (lambda: splittingExactIterativeCN(), 2),
    "splittingExactHeun":          (lambda: splittingExactHeun(), 2),
    "splittingExactMilstein":      (lambda: splittingExactMilstein(), 1),
    "splittingRK4EM":              (lambda: splittingRK4EM(M, N), 4),
    "splittingRK4Milstein":        (lambda: splittingRK4Milstein(M, N), 4),
}


@pytest.fixture
def hamiltonian_system():
    '''System without dissipation (k = 0), so the noise term is zero'''
    return TransmonCavity(M=M, N=N, k=0.0, Omega=1.0, epsilon=0.5, U=2.0)


def exact_solution(system, X0, T):
    '''X(T) = exp(A T) X0, with A the (linear) drift operator built column by column'''
    A = np.zeros((M * N, M * N), dtype=np.complex128)
    BX, bx = np.zeros((M, N), dtype=np.complex128), np.zeros(1, dtype=np.complex128)
    for j in range(M * N):
        e_j = np.zeros(M * N, dtype=np.complex128)
        e_j[j] = 1.0
        system.calculate_drift_matrix(e_j.reshape(M, N), BX, bx, *system.kernel_args())
        A[:, j] = BX.ravel()
    w, V = np.linalg.eig(A)
    return (V @ (np.exp(w * T) * np.linalg.solve(V, X0.ravel()))).reshape(M, N)


def integration_error(make_integrator, system, X0, dt):
    '''Error at time T of the integrator with time step dt (noise z = 0)'''
    integrator, X = make_integrator(), X0.copy()
    BX, ZX = np.zeros_like(X), np.zeros_like(X)
    integrator.precomputations(dt, system)
    for _ in range(round(T / dt)):
        integrator.integrate_step(X, BX, ZX, 0.0, dt, system)
    return np.linalg.norm(X - exact_solution(system, X0, T))


@pytest.mark.parametrize("make_integrator, order", INTEGRATORS.values(), ids=INTEGRATORS.keys())
def test_converges_with_expected_order(make_integrator, order, hamiltonian_system, random_state):
    X0 = random_state(M, N)
    X0 /= np.linalg.norm(X0)
    error_dt = integration_error(make_integrator, hamiltonian_system, X0, dt=0.02)
    error_half_dt = integration_error(make_integrator, hamiltonian_system, X0, dt=0.01)
    measured_order = np.log2(error_dt / error_half_dt)
    assert measured_order == pytest.approx(order, abs=0.3)


def test_taming_scales_drift_in_place():
    '''Tamed drift: BX <- BX / (1 + dt*||BX||), modifying the array passed in'''
    BX = np.ones((2, 2), dtype=np.complex128)        # ||BX|| = 2
    splittingExactEuler.tameDrift(0.5, BX)
    np.testing.assert_allclose(BX, 0.5)
