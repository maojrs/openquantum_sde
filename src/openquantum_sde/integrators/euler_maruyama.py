import numpy as np

from openquantum_sde.integrators.integrator import base_integrator



class EulerMaruyama(base_integrator):
    '''Integrator class for Euler Maruyama '''

    def precomputations(self, dt, system):
        # Make sure conatiners used by integrator are defined in system
        M = system.M
        N = system.N
        system.bx_scalar = np.zeros(1, dtype=np.complex128) 
        

    @staticmethod
    def integrate_step(X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()
        # '''
        system.calculate_drift_matrix(X, BX, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())

        X += dt * BX + z * np.sqrt(dt) * ZX