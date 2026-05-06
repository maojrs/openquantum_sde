import numpy as np
from numba import njit

from openquantum_sde.utils import calculate_norm
from openquantum_sde.integrators import splittingExactEuler



class splittingExactIterativeCN(splittingExactEuler):
    '''Integrator class for splitting method, using exact method for matrix diagonal for
    half time step, then iterative Crank-Nicolson for remaining terms (with Euler-Maruyama 
    noise integration) plus another half time step of the exact solution.'''

    def precomputations(self, dt, system):
            # Make sure conatiners used by integrator are defined in system
            M = system.M
            N = system.N
            system.expdiagBX = np.zeros([M,N], dtype=np.complex128)
            system.bx_scalar = np.zeros(1, dtype=np.complex128)
            system.BXtmp = np.zeros([M,N], dtype=np.complex128) # Used in child integrators
            system.Xk = np.zeros([M,N], dtype=np.complex128) # Used in child integrators

            # Calculate matrix exponentials for exact solution
            system.compute_exponential_drift_matrix_diagonal(system.expdiagBX, 0.5*dt, *system.kernel_args())

    @staticmethod
    def integrate_step_no_taming(X, BX, ZX, z, dt, system):
        """
        Strang splitting + Crank-Nicolson-type correction
        (non-dissipative, much better amplitude behavior)
        """

        sqrt_dt = np.sqrt(dt)

        # 1. Exact diagonal coherent drift solution for half time step
        X *= system.expdiagBX

        # 2. Calculate noise 
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())
        noise = ZX * z * sqrt_dt

        # 3. Nondiagonal drift at current state 
        system.calculate_drift_matrix_nondiagonal(X, system.BXtmp, system.bx_scalar, *system.kernel_args())

        # 4 Predictor (Euler-Maruyama)
        system.Xk = X + dt * system.BXtmp + noise

        # 4. Crank-Nicolson iterations (2 usually enough)
        for _ in range(2):   
            system.calculate_drift_matrix_nondiagonal(system.Xk, BX, system.bx_scalar, *system.kernel_args())

            system.Xk = X + 0.5 * dt * (system.BXtmp + BX) + noise

        # Write back
        X[:] = system.Xk

        # 5. Exact diagonal coherent drift solution for a second half time step
        X *= system.expdiagBX



    def integrate_step_taming(self, X, BX, ZX, z, dt, system):
        """
        Strang splitting + Crank-Nicolson-type correction
        (non-dissipative, much better amplitude behavior)
        """

        sqrt_dt = np.sqrt(dt)

        # 1. Exact diagonal coherent drift solution for half time step
        X *= system.expdiagBX

        # 2. Calculate noise 
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())
        noise = ZX * z * sqrt_dt

        # 3. Nondiagonal drift at current state 
        system.calculate_drift_matrix_nondiagonal(X, system.BXtmp, system.bx_scalar, *system.kernel_args())
        
        # taming
        self.tameDrift(dt, system.BXtmp)

        # 4 Predictor (Euler-Maruyama)
        system.Xk = X + dt * system.BXtmp + noise

        # 4. Crank-Nicolson iterations (2 usually enough)
        for _ in range(2):   
            system.calculate_drift_matrix_nondiagonal(system.Xk, BX, system.bx_scalar, *system.kernel_args())

            # taming
            self.tameDrift(dt, BX)

            system.Xk = X + 0.5 * dt * (system.BXtmp + BX) + noise

        # Write back
        X[:] = system.Xk

        # 5. Exact diagonal coherent drift solution for a second half time step
        X *= system.expdiagBX