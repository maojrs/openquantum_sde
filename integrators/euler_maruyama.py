import numpy as np
from numba import njit


#@njit(fastmath=True)
def euler_maruyama_step(X, BX, ZX, z, dt, system):       
    '''Integrates one step of size dt for an stochastic Schrodinger equation
    with state X. It calculates the total drift matrix BX and the noise matrix 
    ZX, where z is the sample of the complex-valued Wiener noise used to 
    calculate ZX. The specific system with its corresponding parameter should 
    be defined.
    
    # when called *args should be *system.kernel_args()
    # '''
    system.calculate_drift_matrix(X, BX, system.BX_hamiltonian, system.BX_dissipative, system.bx_scalar, *system.kernel_args())
    system.calculate_noise_matrix(X, ZX, z, *system.kernel_args())
    X += dt * BX + np.sqrt(dt) * ZX