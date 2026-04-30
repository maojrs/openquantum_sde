import numpy as np

from openquantum_sde.integrators.integrator import base_integrator


class Milstein(base_integrator):
    '''Integrator class for Milstein '''

    @staticmethod
    def milstein_noise_step(X, dt, ZX, z, system):
        # Step 1: compute B(X) * z
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())

        dW = np.sqrt(dt) * z

        # Euler-Maruyama part
        X += dW * ZX

        # Milstein correction 0.5*(B dot Nabla)(B(x))(dW^2-dt)
        # For linear B(x), then (B dot Nabla)(B(x)) => B(B(x))
        
        # Compute B(B(X)) applied to (dW^2-dt) (elementwise)
        # For real noise Z2 = z * z - 1.0   # since (dW^2 - dt)/dt = (z^2 - 1)
        # For complex noise, we need:
        Z2 = z * z
        
        # apply noise operator again
        system.calculate_noise_matrix(ZX, system.ZXtmp, *system.kernel_args())

        X += 0.5 * dt * system.ZXtmp * Z2


    def integrate_step(self, X, BX, ZX, z, dt, system):       
        '''Integrates one step of size dt for an stochastic Schrodinger equation
        with state X. It calculates the total drift matrix BX and the noise matrix 
        ZX, where z is the sample of the complex-valued Wiener noise used to 
        calculate ZX. The specific system with its corresponding parameter should 
        be defined.
        
        # when called *args should be *system.kernel_args()
        # '''
        system.calculate_drift_matrix(X, BX, system.BX_hamiltonian, system.BX_dissipative, system.bx_scalar, *system.kernel_args())

        # Milstein part (full time-step)
        self.milstein_noise_step(X, dt, ZX, z, system)
        X += dt * BX

