from numba import njit
from openquantum_sde.utils import calculate_norm


@njit(fastmath=True)
def choose_dt_from_drift(BX, dt_min, dt_max, tol, safety) :
    '''Returns a chosen dt depending on the norm of the drift matrix'''
    bnorm = calculate_norm(BX)

    if bnorm == 0.0:
        return dt_max

    dt = safety * tol / bnorm
    if dt < dt_min:
        dt = dt_min
    elif dt > dt_max:
        dt = dt_max
    return dt