"""Tests for the drift and noise routines of the TransmonCavity system.

The optimized routines are checked against the readable ones and against
basic physical properties."""

import numpy as np
import pytest

from openquantum_sde.systems import TransmonCavity


def diagonal_drift(system):
    '''Diagonal part of the drift: i*U/2*m*(m-1) - k*n'''
    m, n = np.ogrid[:system.M, :system.N]
    return 1j * 0.5 * system.U * m * (m - 1.0) - system.k * n


def total_drift(system, X):
    '''Total drift matrix BX and drift scalar bx from calculate_drift_matrix'''
    BX, bx = np.zeros_like(X), np.zeros(1, dtype=np.complex128)
    system.calculate_drift_matrix(X, BX, bx, *system.kernel_args())
    return BX, bx[0]


# =========================
# Drift
# =========================

def test_total_drift_equals_coherent_plus_noncoherent(system, random_state):
    X = random_state(system.M, system.N)
    coherent, noncoherent = np.zeros_like(X), np.zeros_like(X)
    system.drift_matrix_coherent(X, coherent, *system.kernel_args())
    system.drift_matrix_noncoherent(X, noncoherent, np.zeros(1, dtype=np.complex128),
                                    *system.kernel_args())
    np.testing.assert_allclose(total_drift(system, X)[0], coherent + noncoherent, rtol=1e-12)


def test_nondiagonal_drift_plus_diagonal_equals_total_drift(system, random_state):
    X = random_state(system.M, system.N)
    nondiagonal = np.zeros_like(X)
    system.calculate_drift_matrix_nondiagonal(X, nondiagonal, np.zeros(1, dtype=np.complex128),
                                              *system.kernel_args())
    np.testing.assert_allclose(nondiagonal + diagonal_drift(system) * X,
                               total_drift(system, X)[0], rtol=1e-12)


def test_exponential_of_diagonal_drift(system):
    dt = 0.01
    expdiag = np.zeros((system.M, system.N), dtype=np.complex128)
    system.compute_exponential_drift_matrix_diagonal(expdiag, dt, *system.kernel_args())
    np.testing.assert_allclose(expdiag, np.exp(dt * diagonal_drift(system)), rtol=1e-12)


def test_drift_scalar_matches_drift_matrix_routine(system, random_state):
    X = random_state(system.M, system.N)
    bx = np.zeros(1, dtype=np.complex128)
    system.calculate_drift_scalar(X, bx, *system.kernel_args())
    assert bx[0] == pytest.approx(total_drift(system, X)[1], rel=1e-12)


def test_hamiltonian_drift_conserves_norm(random_state):
    '''Without dissipation (k = 0) the drift is anti-Hermitian: Re<X, BX> = 0'''
    system = TransmonCavity(M=3, N=6, k=0.0, Omega=5.0, epsilon=2.0, U=10.0)
    X = random_state(system.M, system.N)
    BX, _ = total_drift(system, X)
    assert abs(np.vdot(X, BX).real) < 1e-12 * np.linalg.norm(X) * np.linalg.norm(BX)


# =========================
# Noise
# =========================

def test_noise_matrix_removes_one_photon(system, random_state):
    '''Noise term is sqrt(2k) * a X, with a the photon annihilation operator:
    ZX[m, n] = sqrt(2k(n+1)) * X[m, n+1], and zero for the highest photon level'''
    X = random_state(system.M, system.N)
    ZX = np.zeros_like(X)
    system.calculate_noise_matrix(X, ZX, *system.kernel_args())
    n = np.arange(system.N - 1)
    np.testing.assert_allclose(ZX[:, :-1], np.sqrt(2 * system.k * (n + 1)) * X[:, 1:], rtol=1e-12)
    assert np.all(ZX[:, -1] == 0)
