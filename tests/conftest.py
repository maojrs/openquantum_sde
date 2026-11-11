"""Shared fixtures for the test suite."""

import numpy as np
import pytest

from openquantum_sde.integrators import (
    EulerMaruyama, Milstein, stochasticHeun,
    splittingExactEuler, splittingExactMidpointEuler, splittingExactIterativeCN,
    splittingExactHeun, splittingExactMilstein, splittingRK4EM, splittingRK4Milstein,
)
from openquantum_sde.systems import TransmonCavity


# All integrators, including the tamed variants (built for a given system size M, N)
ALL_INTEGRATORS = {
    "EulerMaruyama":                     lambda M, N: EulerMaruyama(),
    "Milstein":                          lambda M, N: Milstein(),
    "stochasticHeun":                    lambda M, N: stochasticHeun(),
    "splittingExactEuler":               lambda M, N: splittingExactEuler(),
    "splittingExactEuler_tamed":         lambda M, N: splittingExactEuler(taming=True),
    "splittingExactMidpointEuler":       lambda M, N: splittingExactMidpointEuler(),
    "splittingExactMidpointEuler_tamed": lambda M, N: splittingExactMidpointEuler(taming=True),
    "splittingExactIterativeCN":         lambda M, N: splittingExactIterativeCN(),
    "splittingExactIterativeCN_tamed":   lambda M, N: splittingExactIterativeCN(taming=True),
    "splittingExactHeun":                lambda M, N: splittingExactHeun(),
    "splittingExactHeun_tamed":          lambda M, N: splittingExactHeun(taming=True),
    "splittingExactMilstein":            lambda M, N: splittingExactMilstein(),
    "splittingExactMilstein_tamed":      lambda M, N: splittingExactMilstein(taming=True),
    "splittingRK4EM":                    lambda M, N: splittingRK4EM(M, N),
    "splittingRK4Milstein":              lambda M, N: splittingRK4Milstein(M, N),
}


@pytest.fixture(params=ALL_INTEGRATORS.values(), ids=ALL_INTEGRATORS.keys())
def make_integrator(request):
    '''Factory make_integrator(M, N) for each integrator (tests using it run once per integrator)'''
    return request.param


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
