"""Tests for the utility functions: noise, seeding, matrix operations,
observables and trajectory analysis."""

import numpy as np
import pytest

from openquantum_sde.utils import (
    complex_noise, seed_numba, complex_noise_matrix, diag_matrix_exponential,
    calculate_norm, calculate_num_atoms, calculate_num_photons,
    calculate_num_photons_chunk, filter_trajectory, find_minima_fast,
)


# =========================
# Noise and seeding
# =========================

def test_complex_noise_returns_complex():
    z = complex_noise()
    assert isinstance(z, complex)


def test_complex_noise_statistics():
    '''z = (x + iy)/sqrt(2) with x, y standard normal, so:
    E[z] = 0, E[|z|^2] = 1 and E[z^2] = 0 (needed by the Milstein/Heun schemes)'''
    seed_numba(1)
    z = np.array([complex_noise() for _ in range(20000)])
    assert abs(z.mean()) < 0.03
    assert abs(np.mean(np.abs(z)**2) - 1.0) < 0.05
    assert abs(np.mean(z**2)) < 0.05
    assert abs(np.var(z.real) - 0.5) < 0.03
    assert abs(np.var(z.imag) - 0.5) < 0.03


def test_seed_numba_reproducible():
    seed_numba(42)
    a = [complex_noise() for _ in range(10)]
    seed_numba(42)
    b = [complex_noise() for _ in range(10)]
    seed_numba(43)
    c = [complex_noise() for _ in range(10)]
    assert a == b
    assert a != c


def test_complex_noise_matrix_fills_and_statistics():
    seed_numba(2)
    Z = np.zeros((100, 200), dtype=np.complex128)
    complex_noise_matrix(Z)
    assert np.all(Z != 0)
    assert abs(Z.mean()) < 0.03
    assert abs(np.mean(np.abs(Z)**2) - 1.0) < 0.05
    assert abs(np.mean(Z**2)) < 0.05


def test_complex_noise_matrix_reproducible():
    Z1 = np.zeros((4, 5), dtype=np.complex128)
    Z2 = np.zeros((4, 5), dtype=np.complex128)
    seed_numba(7); complex_noise_matrix(Z1)
    seed_numba(7); complex_noise_matrix(Z2)
    np.testing.assert_array_equal(Z1, Z2)


# =========================
# Matrix operations
# =========================

def test_diag_matrix_exponential_matches_numpy(random_state):
    D = random_state(3, 4)
    expD = np.zeros_like(D)
    diag_matrix_exponential(3, 4, D, expD)
    np.testing.assert_allclose(expD, np.exp(D), rtol=1e-12)


def test_calculate_norm_matches_numpy(random_state):
    X = random_state(5, 7)
    assert calculate_norm(X) == pytest.approx(np.linalg.norm(X), rel=1e-12)


def test_calculate_norm_zero_and_unit():
    assert calculate_norm(np.zeros((3, 3), dtype=np.complex128)) == 0.0
    X = np.zeros((3, 3), dtype=np.complex128)
    X[1, 2] = 1j
    assert calculate_norm(X) == pytest.approx(1.0)


# =========================
# Observables
# =========================

def test_calculate_num_atoms_distribution(random_state):
    '''Returns the probability of each transmon level m (sums to 1)'''
    X = random_state(4, 6)
    p = calculate_num_atoms(X)
    expected = np.sum(np.abs(X)**2, axis=1) / np.sum(np.abs(X)**2)
    np.testing.assert_allclose(p, expected, rtol=1e-12)
    assert p.sum() == pytest.approx(1.0)


def test_calculate_num_atoms_single_level():
    X = np.zeros((4, 6), dtype=np.complex128)
    X[2, 3] = 0.5 + 0.5j
    np.testing.assert_allclose(calculate_num_atoms(X), [0, 0, 1, 0])


def test_calculate_num_photons_fock_state():
    '''State with exactly n = 3 photons gives <n> = 3, whatever the
    transmon level, phase or normalization'''
    X = np.zeros((3, 6), dtype=np.complex128)
    X[1, 3] = 2.0 * np.exp(0.7j)
    n = calculate_num_photons(X)
    assert n.real == pytest.approx(3.0)
    assert abs(n.imag) < 1e-12


def test_calculate_num_photons_superposition_and_scaling(random_state):
    X = random_state(3, 8)
    probs = np.abs(X)**2
    expected = np.sum(probs * np.arange(8)[None, :]) / probs.sum()
    assert calculate_num_photons(X).real == pytest.approx(expected, rel=1e-12)
    # Independent of normalization
    assert calculate_num_photons(5.0 * X).real == pytest.approx(expected, rel=1e-12)


def test_calculate_num_photons_chunk_matches_numpy():
    rng = np.random.default_rng(3)
    traj = rng.standard_normal((10, 3, 6)) + 1j * rng.standard_normal((10, 3, 6))
    total_photons, total_norm = calculate_num_photons_chunk(traj)
    probs = np.abs(traj)**2
    assert total_norm == pytest.approx(probs.sum(), rel=1e-12)
    assert total_photons == pytest.approx(np.sum(probs * np.arange(6)), rel=1e-12)


def test_calculate_num_photons_chunk_single_step_matches_num_photons(random_state):
    X = random_state(3, 6)
    total_photons, total_norm = calculate_num_photons_chunk(X[None, :, :])
    assert total_photons / total_norm == pytest.approx(calculate_num_photons(X).real, rel=1e-12)


# =========================
# Trajectory analysis
# =========================

def test_filter_trajectory_selects_points_near_minima():
    traj = np.arange(5 * 2 * 2).reshape(5, 2, 2)
    traj_current = np.array([0.0, 0.1, 5.0, 5.1, 0.05j])
    filtered = filter_trajectory(traj, traj_current, minima=0.0, tolerance=0.2)
    np.testing.assert_array_equal(filtered, traj[[0, 1, 4]])


def test_filter_trajectory_tolerance_is_inclusive():
    traj = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    traj_current = np.array([1.0 + 0j, 0.5j, 2.0])
    filtered = filter_trajectory(traj, traj_current, minima=1.0j, tolerance=0.5)
    np.testing.assert_array_equal(filtered, traj[[1]])


def test_find_minima_fast_two_clusters():
    rng = np.random.default_rng(4)
    c1, c2 = 0.0 + 0.0j, 10.0 + 5.0j
    pts = np.concatenate([
        c1 + 0.1 * (rng.standard_normal(500) + 1j * rng.standard_normal(500)),
        c2 + 0.1 * (rng.standard_normal(500) + 1j * rng.standard_normal(500)),
    ])
    minimas = find_minima_fast(pts, bins=5)
    assert len(minimas) >= 2
    # Every returned center lies within one bin of a true cluster center
    bin_size = abs(np.ptp(pts.real)) / 5 + abs(np.ptp(pts.imag)) / 5
    for m in minimas:
        assert min(abs(m - c1), abs(m - c2)) < bin_size
    # Both clusters are found
    assert min(abs(minimas - c1)) < bin_size
    assert min(abs(minimas - c2)) < bin_size
