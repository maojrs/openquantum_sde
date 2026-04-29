import numpy as np

from openquantum_sde.integrators import splittingRK4EM



class splittingRK4Milstein(splittingRK4EM):
    '''Integrator class for a spliting method using Runge-Kutta 4 for the
    drift part (half time step), then Milstein for the stochastic part (full
    timstep) and then again RK4 for another half-time step. The RK4 routines
    are inherited from the parent class.'''


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
        Z2 = z * z - 1.0   # since (dW^2 - dt)/dt = (z^2 - 1)
        
        # apply noise operator again
        system.calculate_noise_matrix(ZX, system.ZX2, *system.kernel_args())

        X += 0.5 * dt * system.ZX2 * Z2


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
        self.milstein_noise_step(X, dt, ZX, z, system)

        # Second half-time step RK4        
        self.rk4_drift_step(X, 0.5*dt, BX, system)

        system.calculate_drift_matrix(X, BX, 
                                system.BX_hamiltonian, system.BX_dissipative, system.bx_scalar, 
                                *system.kernel_args())


