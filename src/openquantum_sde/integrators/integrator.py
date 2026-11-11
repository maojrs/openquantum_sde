from numba import njit


class base_integrator:
    """Abstract base class for systentegrators of stochastic Schrodinger equations.
    The child class should define at least one function called integrate_step."""

    def __init__(self):
        pass


    @staticmethod
    @njit(fastmath = True)
    def euler_substep(out, X, a, K):
        '''Euler sub-step: out = X + a*K.
        Written into the existing array out (no new array allocated).
        out may be the same array as X.'''
        M, N = X.shape
        for m in range(M):
            for n in range(N):
                out[m, n] = X[m, n] + a * K[m, n]


    def precomputations(self, dt, system):
        pass


    def recomputations_newdt(self, dt, system):
        pass


    def integrate_step(self, X, BX, ZX, z, dt, system):
        raise NotImplementedError
    