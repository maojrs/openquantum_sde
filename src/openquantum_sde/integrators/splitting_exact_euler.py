import numpy as np
from numba import njit

from openquantum_sde.utils import calculate_norm
from openquantum_sde.integrators.integrator import base_integrator


class splittingExactEuler(base_integrator):
    '''Integrator class for splitting method, using exact method for matrix diagonal for
     half time step, then Euler method for remaining terms for a full time step plus another
    half time step of the exact solution.

    Warning: the drift is advanced with an explicit Euler step, which slightly
    amplifies undamped oscillations at every step. For weakly damped, strongly
    coupled systems (e.g. large Omega with an undamped transmon) the solution can
    blow up unless dt is very small. A second-order scheme such as
    splittingExactIterativeCN (or splittingExactHeun, splittingRK4EM) is more robust.'''

    def __init__(self, taming = False):
        super().__init__()

        if taming:
            self.integrate_step = self.integrate_step_taming
        else:
            self.integrate_step = self.integrate_step_no_taming


    def precomputations(self, dt, system):
        # Make sure conatiners used by integrator are defined in system
        M = system.M
        N = system.N
        system.expdiagBX = np.zeros([M,N], dtype=np.complex128)
        system.bx_scalar = np.zeros(1, dtype=np.complex128)

        # Calculate matrix exponentials for exact solution
        system.compute_exponential_drift_matrix_diagonal(system.expdiagBX, 0.5*dt, *system.kernel_args())

        

    def recomputations_newdt(self, dt, system):
        # If time step is changed, updated matrix exponentials for exact solution
        system.compute_exponential_drift_matrix_diagonal(system.expdiagBX, 0.5*dt, *system.kernel_args())
    

    @staticmethod
    @njit(fastmath = True)
    def tameDrift(dt, BX):
        '''Tames the drift in place: BX <- BX / (1 + dt*||BX||)'''
        normBX = calculate_norm(BX)
        factor = 1.0 / (1.0 + dt * normBX)
        M, N = BX.shape
        for m in range(M):
            for n in range(N):
                BX[m, n] *= factor



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

        # Euler method for remaining terms
        system.calculate_drift_matrix_nondiagonal(X, BX, system.bx_scalar, *system.kernel_args())
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

        # Euler method for remaining terms
        system.calculate_drift_matrix_nondiagonal(X, BX, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())

        #Taming
        self.tameDrift(dt, BX)

        X += BX * dt + ZX * z * np.sqrt(dt) 
        
        # Exact diagonal coherent drift solution for a second half time step
        X *= system.expdiagBX