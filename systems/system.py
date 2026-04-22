

class base_system:
    """Abstract base class for systems with stochastic Schrodinger equations."""
    
    def kernel_args(self):
        raise NotImplementedError
    
