import numpy as np

from openquantum_sde.integrators.splitting_exact_euler import splittingExactEuler
from openquantum_sde.integrators.milstein import Milstein



class splittingExactMilstein(splittingExactEuler, Milstein):
    '''Integrator class for splitting method, using exact method for matrix diagonal for
     half time step, then Milstein method for remaining terms for a full time step plus another
    half time step of the exact solution.

    Warning: the drift is advanced with an explicit Euler step, which slightly
    amplifies undamped oscillations at every step. For weakly damped, strongly
    coupled systems (e.g. large Omega with an undamped transmon) the solution can
    blow up unless dt is very small. A second-order scheme such as
    splittingExactIterativeCN (or splittingExactHeun, splittingRK4EM) is more robust.'''
    

    def precomputations(self, dt, system):
        # Make sure conatiners used by integrator are defined in system
        M = system.M
        N = system.N
        system.expdiagBX = np.zeros([M,N], dtype=np.complex128)
        system.bx_scalar = np.zeros(1, dtype=np.complex128)
        system.ZXtmp = np.zeros([M,N], dtype=np.complex128)


        # Calculate matrix exponentials for exact solution
        system.compute_exponential_drift_matrix_diagonal(system.expdiagBX, 0.5*dt, *system.kernel_args())


    def integrate_step_no_taming(self, X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()'''

        # Exact diagonal coherent drift solution for half time step
        X *= system.expdiagBX

        # Milstein method for remaining terms

        system.calculate_drift_matrix_nondiagonal(X, BX, system.bx_scalar, *system.kernel_args())

        # Milstein part (full time-step)
        self.milstein_noise_step(X, dt, ZX, z, system)
        X += dt * BX

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

        # Milstein method with Taming for remaining terms
   
        system.calculate_drift_matrix_nondiagonal(X, BX, system.bx_scalar, *system.kernel_args())

        # Taming
        self.tameDrift(dt, BX)

        # Milstein part (full time-step)
        self.milstein_noise_step(X, dt, ZX, z, system)
        X += dt * BX
        
        # Exact diagonal coherent drift solution for a second half time step
        X *= system.expdiagBX