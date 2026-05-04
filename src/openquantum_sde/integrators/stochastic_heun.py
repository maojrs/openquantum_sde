import numpy as np

from openquantum_sde.integrators.integrator import base_integrator



class stochasticHeun(base_integrator):
    '''Integrator class for stochastic Heun method (predictor-corrector) '''

    
    def integrate_step(self, X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()
        # '''
        system.calculate_drift_matrix(X, system.BXtmp, system.BX_coherent, system.BX_noncoherent, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(X, system.ZXtmp, *system.kernel_args())

        predictor = X + dt * system.BXtmp + z * np.sqrt(dt) * system.ZXtmp 

        system.calculate_drift_matrix(predictor, BX, system.BX_coherent, system.BX_noncoherent, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(predictor, ZX, *system.kernel_args())
        
        
        X += 0.5 * (system.BXtmp + BX) * dt + 0.5 * (system.ZXtmp + ZX) * z * np.sqrt(dt) 