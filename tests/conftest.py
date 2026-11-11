"""Shared fixtures for the test suite."""

import numpy as np
import pytest

from openquantum_sde.systems import TransmonCavity


@pytest.fixture
def random_state():
    '''Factory for random complex (not normalized) states of shape (M, N)'''
    rng = np.random.default_rng(0)
    return lambda M, N: rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))


@pytest.fixture
def system():
    '''Small transmon-cavity system with dissipation (k > 0)'''
    return TransmonCavity(M=3, N=6, k=1.0, Omega=5.0, epsilon=2.0, U=10.0)


@pytest.fixture
def vacuum(system):
    '''Initial state with no transmon excitation and no photons'''
    X0 = np.zeros((system.M, system.N), dtype=np.complex128)
    X0[0, 0] = 1.0
    return X0
