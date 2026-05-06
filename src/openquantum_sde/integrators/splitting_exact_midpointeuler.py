import numpy as np
from numba import njit

from openquantum_sde.utils import calculate_norm
from openquantum_sde.integrators import splittingExactEuler



class splittingExactMidpointEuler(splittingExactEuler):
    '''Integrator class for splitting method, using exact method for matrix diagonal for
    half time step, then mid-point Euler method for drift (Euler-Maruyama for noise), 
    then remaining terms for a full time step plus another half time step of the exact solution.'''
    

    def precomputations(self, dt, system):
        # Make sure conatiners used by integrator are defined in system
        M = system.M
        N = system.N
        system.expdiagBX = np.zeros([M,N], dtype=np.complex128)
        system.bx_scalar = np.zeros(1, dtype=np.complex128)
        system.BXtmp = np.zeros([M,N], dtype=np.complex128) 

        # Calculate matrix exponentials for exact solution
        system.compute_exponential_drift_matrix_diagonal(system.expdiagBX, 0.5*dt, *system.kernel_args())


    @staticmethod
    def integrate_step_no_taming(X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()'''

        # Exact diagonal coherent drift solution for half time step
        X *= system.expdiagBX

        # Semi-implicit Euler method for remaining terms
        system.calculate_drift_matrix_nondiagonal(X, system.BXtmp, system.bx_scalar, *system.kernel_args())

        predictor = X + system.BXtmp * 0.5 * dt

        # One iteration
        system.calculate_drift_matrix_nondiagonal(predictor, BX, system.bx_scalar, *system.kernel_args())

        # Calculate noise
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())

        X += BX * dt + ZX * z * np.sqrt(dt) 

        # Exact diagonal coherent drift solution for a second half time step
        X *= system.expdiagBX


    def integrate_step_taming(self, X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X with taming. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()'''

        # Exact diagonal coherent drift solution for half time step
        X *= system.expdiagBX

        # Semi-implicit Euler method for remaining terms
        system.calculate_drift_matrix_nondiagonal(X, system.BXtmp, system.bx_scalar, *system.kernel_args())

        #Taming
        self.tameDrift(dt, system.BXtmp)

        predictor = X + system.BXtmp * 0.5 * dt 

        # One iteration
        system.calculate_drift_matrix_nondiagonal(predictor, BX, system.bx_scalar, *system.kernel_args())

        #Taming
        self.tameDrift(dt, BX)

        # Calculate noise
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())

        X += BX * dt + ZX * z * np.sqrt(dt) 
        
        # Exact diagonal coherent drift solution for a second half time step
        X *= system.expdiagBX