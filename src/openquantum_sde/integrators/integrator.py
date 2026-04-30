

class base_integrator:
    """Abstract base class for systentegrators of stochastic Schrodinger equations.
    The child class should define at least one function called integrate_step."""

    def precomputations(self, dt, system):
        pass


    def recomputations_newdt(self, dt, system):
        pass


    def integrate_step(self, X, BX, ZX, z, dt, system):
        raise NotImplementedError
    