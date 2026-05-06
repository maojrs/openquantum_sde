import numpy as np

from openquantum_sde.integrators import splittingRK4EM, Milstein



class splittingRK4Milstein(splittingRK4EM, Milstein):
    '''Integrator class for a spliting method using Runge-Kutta 4 for the
    drift part (half time step), then Milstein for the stochastic part (full
    timstep) and then again RK4 for another half-time step. The RK4 routines
    are inherited from the parent class.'''


    def precomputations(self, dt, system):
        # Make sure conatiners used by integrator are defined in system
        M = system.M
        N = system.N
        system.bx_scalar = np.zeros(1, dtype=np.complex128)
        system.ZXtmp = np.zeros([M,N], dtype=np.complex128)


    def integrate_step(self, X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()
        '''

        # Operator splitting with RK4, EM, RK4

        # First half-time step RK4
        self.rk4_drift_step(X, 0.5*dt, BX, system)
        
        # Milstein part (full time-step)
        self.milstein_noise_step(X, dt, ZX, z, system)

        # Second half-time step RK4        
        self.rk4_drift_step(X, 0.5*dt, BX, system)


