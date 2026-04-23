

class base_system:
    """Abstract base class for systems with stochastic Schrodinger equations.
    The child class should define at least these three functions (perhaps with 
    additional inputs). The decorators are recommended for improved speed with 
    numba, but then self must not be used."""
    
    def kernel_args(self):
        raise NotImplementedError
    
    #@staticmethod
    #@njit(fastmath=True)
    def calculate_drift_matrix(X, BX):
        raise NotImplementedError
    
    #@staticmethod
    #@njit(fastmath=True)
    def calculate_noise_matrix(X, ZX):
        raise NotImplementedError

    
