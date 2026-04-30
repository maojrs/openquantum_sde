import numpy as np
from numba import njit

from openquantum_sde.utils import calculate_norm
from openquantum_sde.integrators.integrator import base_integrator


class splittingExactEuler(base_integrator):
    '''Integrator class for splitting method, using exact method for matrix diagonal for
     half time step, then Euler method for remaining terms for a full time step plus another
    half time step of the exact solution.'''

    def __init__(self, taming = False):
        if taming:
            self.integrate_step = self.integrate_step_taming
        else:
            self.integrate_step = self.integrate_step_no_taming


    def precomputations(self, dt, system):
        # Calculate matrix exponentials for exact solution
        system.compute_exponential_drift_matrix_diagonal(system.expdiagBX, 0.5*dt, *system.kernel_args())


    def recomputations_newdt(self, dt, system):
        # If time step is changed, updated matrix exponentials for exact solution
        system.compute_exponential_drift_matrix_diagonal(system.expdiagBX, 0.5*dt, *system.kernel_args())
    

    @staticmethod
    @njit(fastmath = True)
    def tameDrift(dt, BX):
        normBX = calculate_norm(BX)
        BX = BX / (1 + dt * normBX)



    @staticmethod
    def integrate_step_no_taming(X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()'''

        # Exact diagonal drift solution for half time step
        X *= system.expdiagBX

        # Euler method for remaining terms
        system.calculate_drift_matrix_nondiagonal(X, BX, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())

        X += BX * dt + ZX * z * np.sqrt(dt) 

        # Exact diagonal drift solution for half time step
        X *= system.expdiagBX


    def integrate_step_taming(self, X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X with taming. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()'''

        # Exact diagonal drift solution for half time step
        X *= system.expdiagBX

        # Euler method for remaining terms
        system.calculate_drift_matrix_nondiagonal(X, BX, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())

        #Taming
        self.tameDrift(dt, BX)

        X += BX * dt + ZX * z * np.sqrt(dt) 
        
        # Exact diagonal drift solution for half time step
        X *= system.expdiagBX