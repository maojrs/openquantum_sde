import numpy as np
from tqdm import tqdm

from openquantum_sde.utils import calculate_norm, complex_noise, seed_numba
from openquantum_sde.integrators.time_adaptive import choose_dt_from_drift


# Safety factor for the adaptive time step
SAFETY = 0.9


def maybe_tqdm(iterable, use_tqdm, tqdm_kwargs=None, **kwargs):
    '''For correct display of progres bar (serial and parallelized siulations)'''
    if not use_tqdm:
        return iterable

    if tqdm_kwargs is None:
        tqdm_kwargs = {"desc": "Simulating"}

    return tqdm(iterable, **kwargs, **tqdm_kwargs)


def simulate_fixed_dt(X0, nsteps, dt,
                      calculate_current=False,
                      save_every=1, renormalize_every=1,
                      progress_bar=True, tqdm_kwargs=None,
                      integrator=None, system=None, seed=None):
    '''Simulates nsteps steps of size dt from the initial state X0.
    Returns dt_array, time_array, traj, traj_current (one entry every save_every steps).'''
    return _simulate(X0, nsteps, dt, None, calculate_current, save_every, renormalize_every,
                     progress_bar, tqdm_kwargs, integrator, system, seed)


def simulate_adaptive_dt(X0, nsteps_max, dt_min, dt_max, tol,
                         calculate_current=False,
                         save_every=1, renormalize_every=1,
                         progress_bar=True, tqdm_kwargs=None,
                         integrator=None, system=None, seed=None):
    '''Simulates nsteps_max steps from the initial state X0, choosing dt in [dt_min, dt_max]
    from the drift every renormalize_every steps (see choose_dt_from_drift).
    Returns dt_array, time_array, traj, traj_current (one entry every save_every steps).'''
    return _simulate(X0, nsteps_max, dt_min, (dt_min, dt_max, tol), calculate_current, save_every,
                     renormalize_every, progress_bar, tqdm_kwargs, integrator, system, seed)


def _simulate(X0, nsteps, dt, dt_limits, calculate_current, save_every, renormalize_every,
              progress_bar, tqdm_kwargs, integrator, system, seed):
    '''Shared time loop. dt stays fixed if dt_limits is None, otherwise it is
    chosen from the drift BX after each renormalization, with
    dt_limits = (dt_min, dt_max, tol).'''

    if system is None or integrator is None:
        raise ValueError("simulate needs both an integrator and a system")

    if seed is not None:
        seed_numba(seed)

    # Initialize arrays and auxiliary system containers
    X = np.asarray(X0, dtype=np.complex128).copy()
    BX = np.zeros_like(X)
    ZX = np.zeros_like(X)
    system.bx_scalar = np.zeros(1, dtype=np.complex128)

    # Output arrays
    nsave = nsteps // save_every + 1
    traj = np.empty((nsave, system.M, system.N), dtype=np.complex128)
    time_array = np.empty(nsave, dtype=np.float64)
    traj_current = np.zeros(nsave, dtype=np.complex128)
    traj[0] = X
    time_array[0] = 0.0
    save_idx = 1

    # Initial time and current
    t = 0.0
    alpha = 0.0 + 0.0j

    # Run precomputations that might be needed by integrator
    integrator.precomputations(dt, system)

    for step in maybe_tqdm(range(1, nsteps + 1), progress_bar, tqdm_kwargs=tqdm_kwargs):

        # Main integrator step
        z = complex_noise()
        integrator.integrate_step(X, BX, ZX, z, dt, system)
        t += dt

        # Calculate current (drift scalar recalculated at the end of the step)
        if calculate_current:
            system.calculate_drift_scalar(X, system.bx_scalar, *system.kernel_args())
            alpha = system.backward_euler_step_current(alpha, z, dt, system.bx_scalar, system.k, system.kfill)

        # Renormalize and, if adaptive, choose new dt from the drift of the last step
        if step % renormalize_every == 0:
            X /= calculate_norm(X)
            if dt_limits is not None:
                dt = choose_dt_from_drift(BX, *dt_limits, SAFETY)
                integrator.recomputations_newdt(dt, system)

        # Save data into arrays
        if step % save_every == 0:
            traj[save_idx] = X
            time_array[save_idx] = t
            traj_current[save_idx] = alpha
            save_idx += 1

    # Array of time steps (averaged over save_every steps)
    dt_array = np.diff(time_array) / save_every

    return dt_array[:save_idx-1], time_array[:save_idx], traj[:save_idx], traj_current[:save_idx]
