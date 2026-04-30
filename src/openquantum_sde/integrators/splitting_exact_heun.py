import numpy as np

from openquantum_sde.integrators import splittingExactEuler



class splittingExactHeun(splittingExactEuler):
    '''Integrator class for splitting method, using exact method for matrix diagonal for
     half time step, then Huen method for remaining terms for a full time step plus another
    half time step of the exact solution.'''

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

        # Heun method for remaining terms

        system.calculate_drift_matrix_nondiagonal(X, system.BXtmp, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(X, system.ZXtmp, *system.kernel_args())

        predictor = X + dt * BX + z * np.sqrt(dt) * ZX 

        system.calculate_drift_matrix_nondiagonal(predictor, BX, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(predictor, ZX, *system.kernel_args())
        
        X += 0.5 * (system.BXtmp + BX) * dt + 0.5 * (system.ZXtmp + ZX) * z * np.sqrt(dt) 

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

        # Heun method with Taming for remaining terms

        system.calculate_drift_matrix_nondiagonal(X, system.BXtmp, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(X, system.ZXtmp, *system.kernel_args())
    
        self.tameDrift(dt, system.BXtmp)

        predictor = X + dt * BX + z * np.sqrt(dt) * ZX

        system.calculate_drift_matrix_nondiagonal(predictor, BX, system.bx_scalar, *system.kernel_args())
        system.calculate_noise_matrix(predictor, ZX, *system.kernel_args())

        self.tameDrift(dt, BX)
        
        X += 0.5 * (system.BXtmp + BX) * dt + 0.5 * (system.ZXtmp + ZX) * z * np.sqrt(dt) 

        
        # Exact diagonal drift solution for half time step
        X *= system.expdiagBX