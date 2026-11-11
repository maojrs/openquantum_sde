"""Tests that every name listed in a package's __all__ can actually be imported."""

import pytest

import openquantum_sde
from openquantum_sde import integrators, systems


@pytest.mark.parametrize("package", [openquantum_sde, integrators, systems],
                         ids=lambda p: p.__name__)
def test_all_listed_names_are_importable(package):
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []
