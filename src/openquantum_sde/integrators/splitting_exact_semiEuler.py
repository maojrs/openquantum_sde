import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from numba import njit

from openquantum_sde.utils import calculate_norm
from openquantum_sde.integrators import splittingExactEuler



class splittingExactSemiImplicitEuler(splittingExactEuler):
    '''Integrator class for splitting method, using exact method for matrix diagonal for
    half time step, then semi-implicit Euler method for remaining terms plus another half 
    time step of the exact solution.'''


    def __init__(self): 
        pass


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

        # Semi-implicit Euler method for remaining terms
        system.calculate_drift_matrix_nondiagonal(X, system.BXtmp, system.bx_scalar, *system.kernel_args())

        predictor = X + system.BXtmp * 0.5 * dt

        # One iteration
        system.calculate_drift_matrix_nondiagonal(predictor, BX, system.bx_scalar, *system.kernel_args())

        # Calculate noise
        system.calculate_noise_matrix(X, ZX, *system.kernel_args())

        X += BX * dt + ZX * z * np.sqrt(dt) 

        # Exact diagonal drift solution for half time step
        X *= system.expdiagBX



@staticmethod
def integrate_step(X, BX, ZX, z, dt, system):
    """
    Strang splitting + implicit Euler (non-diagonal drift)
    solved via predictor + damped Newton-Krylov.
    """

    sqrt_dt = np.sqrt(dt)

    # --- 1. Half-step exact diagonal ---
    X *= system.expdiagBX

    # --- 2. Noise evaluated at current state ---
    system.calculate_noise_matrix(X, ZX, *system.kernel_args())
    noise = ZX * z * sqrt_dt

    # Build B_n
    Bn = X + noise

    # --- 3. Predictor (explicit EM) ---
    system.calculate_drift_matrix_nondiagonal(
        X, BX, system.bx_scalar, *system.kernel_args()
    )
    Xk = X + dt * BX + noise

    # Flatten helpers (for GMRES)
    shape = X.shape
    size = X.size

    def flatten(A):
        return A.reshape(size)

    def unflatten(v):
        return v.reshape(shape)

    # --- Residual function ---
    def residual(Xvec):
        Xmat = unflatten(Xvec)

        system.calculate_drift_matrix_nondiagonal(
            Xmat, BX, system.bx_scalar, *system.kernel_args()
        )

        F = Xmat - dt * BX - Bn
        return flatten(F)

    # --- Jacobian-vector product (finite difference) ---
    def Jv(Xvec, v):
        eps = 1e-7

        Xmat = unflatten(Xvec)
        Vmat = unflatten(v)

        system.calculate_drift_matrix_nondiagonal(
            Xmat, BX, system.bx_scalar, *system.kernel_args()
        )
        f0 = BX.copy()

        system.calculate_drift_matrix_nondiagonal(
            Xmat + eps * Vmat, ZX, system.bx_scalar, *system.kernel_args()
        )
        f1 = ZX.copy()

        Jf_v = (f1 - f0) / eps

        return flatten(Vmat - dt * Jf_v)

    # --- Newton iterations ---
    Xvec = flatten(Xk)

    for _ in range(4):  # 3–5 iterations typical

        F = residual(Xvec)
        normF = np.linalg.norm(F)

        if normF < 1e-8:
            break

        # Linear operator for GMRES
        A = LinearOperator((size, size), matvec=lambda v: Jv(Xvec, v))

        # Solve J dX = -F
        dX, _ = gmres(A, -F, tol=1e-6, maxiter=20)

        # --- Line search damping (CRUCIAL for decay stability) ---
        alpha = 1.0
        for _ in range(5):
            Xtrial = Xvec + alpha * dX
            if np.linalg.norm(residual(Xtrial)) < normF:
                break
            alpha *= 0.5

        Xvec = Xvec + alpha * dX

    # Write back
    X[:] = unflatten(Xvec)

    # --- 4. Final half-step diagonal ---
    X *= system.expdiagBX