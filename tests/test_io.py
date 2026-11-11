"""Tests for saving and loading trajectories (HDF5) and parameters (JSON)."""

import numpy as np
import pytest

from openquantum_sde.io import (
    save_trajectory, load_trajectory, load_trajectory_chunk,
    load_trajectory_chunk_random_start, load_trajectory_random_chunk,
    save_params, load_params,
)


@pytest.fixture
def saved_trajectory(tmp_path):
    '''Random trajectory saved to a temporary folder: (folder, times, traj, current)'''
    rng = np.random.default_rng(0)
    times = np.linspace(0, 1, 20)
    traj = rng.standard_normal((20, 2, 3)) + 1j * rng.standard_normal((20, 2, 3))
    current = rng.standard_normal(20) + 1j * rng.standard_normal(20)
    save_trajectory("traj", tmp_path, times, traj, current, simid="test")
    return tmp_path, times, traj, current


def row_indices(chunk, traj):
    '''Index in traj of each time step in chunk (rows are random, so unique)'''
    return [int(np.flatnonzero(traj[:, 0, 0] == row[0, 0])[0]) for row in chunk]


# =========================
# Trajectories
# =========================

def test_save_and_load_trajectory_roundtrip(saved_trajectory):
    folder, *saved = saved_trajectory
    for loaded, original in zip(load_trajectory("traj", folder), saved):
        np.testing.assert_array_equal(loaded, original)


def test_save_trajectory_without_simid(saved_trajectory):
    folder, *saved = saved_trajectory
    save_trajectory("no_simid", folder, *saved)
    for loaded, original in zip(load_trajectory("no_simid", folder), saved):
        np.testing.assert_array_equal(loaded, original)


def test_load_trajectory_chunk_returns_first_steps(saved_trajectory):
    folder, _, traj, _ = saved_trajectory
    np.testing.assert_array_equal(load_trajectory_chunk("traj", folder, 5), traj[:5])


def test_load_trajectory_chunk_random_start_returns_consecutive_steps(saved_trajectory):
    folder, _, traj, _ = saved_trajectory
    chunk = load_trajectory_chunk_random_start("traj", folder, 5)
    start = row_indices(chunk, traj)[0]
    np.testing.assert_array_equal(chunk, traj[start:start + 5])


def test_load_trajectory_random_chunk_returns_distinct_steps_in_order(saved_trajectory):
    folder, _, traj, _ = saved_trajectory
    indices = row_indices(load_trajectory_random_chunk("traj", folder, 5), traj)
    assert len(indices) == 5
    assert indices == sorted(set(indices))


# =========================
# Parameters
# =========================

def test_save_and_load_params_roundtrip(tmp_path):
    params = {"M": 21, "N": 321, "k": 1.0, "epsilon": 10.0, "integrator": "splittingExactIterativeCN"}
    save_params("params.json", tmp_path, params)
    assert load_params("params.json", tmp_path) == params
