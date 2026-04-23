from openquantum_sde.utils import complex_noise

def test_complex_noise():
    z = complex_noise()
    assert isinstance(z, complex)
