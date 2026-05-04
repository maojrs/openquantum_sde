import numpy as np

from openquantum_sde.integrators.integrator import base_integrator



class splittingRK4EM(base_integrator):
    '''Integrator class for a spliting method using Runge-Kutta 4 for the
    drift part (half time step), then Euler Maruyama for the stochastic part (full
    timstep) and then again RK4 for another half-time step'''

    def __init__(self, M, N):
        # Allocate containers for RK4 scheme
        self.K1 = np.zeros([M,N], dtype=np.complex128)
        self.K2 = np.zeros([M,N], dtype=np.complex128)
        self.K3 = np.zeros([M,N], dtype=np.complex128)
        self.K4 = np.zeros([M,N], dtype=np.complex128)
        self.TMP = np.zeros([M,N], dtype=np.complex128)

        

    def rk4_drift_step(self, X, dt, BX, system):


        # ---- K1 (BX) ----
        system.calculate_drift_matrix(X, self.K1, 
                                      system.BX_coherent, system.BX_noncoherent, system.bx_scalar, 
                                      *system.kernel_args())
        self.TMP = X + 0.5 * dt * self.K1

        # ---- K2 ----
        system.calculate_drift_matrix(self.TMP, self.K2, 
                                      system.BX_coherent, system.BX_noncoherent, system.bx_scalar, 
                                      *system.kernel_args())
        self.TMP = X + 0.5 * dt * self.K2

        # ---- K3 ----
        system.calculate_drift_matrix(self.TMP, self.K3, 
                                      system.BX_coherent, system.BX_noncoherent, system.bx_scalar,
                                      *system.kernel_args())
        self.TMP = X + dt * self.K3

        # ---- K4 ----
        system.calculate_drift_matrix(self.TMP, self.K4, 
                                      system.BX_coherent, system.BX_noncoherent, system.bx_scalar,
                                      *system.kernel_args())

        # ---- final update ----
        X += (dt / 6.0) * (self.K1 + 2*self.K2 + 2*self.K3 + self.K4)


    @staticmethod
    def em_noise_step(X, dt, ZX, z, system):
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())
        X += z * np.sqrt(dt) * ZX


    def integrate_step(self, X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()
        # '''

        # Operator splitting with RK4, EM, RK4

        # First half-time step RK4
        self.rk4_drift_step(X, 0.5*dt, BX, system)
        
        # Euler-Maruyama part (full time-step)
        self.em_noise_step(X, dt, ZX, z, system)

        # Second half-time step RK4        
        self.rk4_drift_step(X, 0.5*dt, BX, system)


