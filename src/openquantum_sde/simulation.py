import numpy as np
from tqdm import tqdm
from numba import njit

from openquantum_sde.utils import calculate_norm, complex_noise
from openquantum_sde.integrators.time_adaptive import choose_dt_from_drift


def maybe_tqdm(iterable, use_tqdm, **kwargs):
    ''' Wrapper for loops with percentage bar.'''
    return tqdm(iterable, **kwargs) if use_tqdm else iterable


def simulate_fixed_dt(X0, nsteps, dt, 
                      save_every=1, progress_bar=True, calculate_current=False, 
                      integrator=None, system=None):

    if system == None or integrator == None:
        raise Exception("Need to specify integrator and/or system and system.kernel_args()")

    print_every = max(1, nsteps // 20)   # 5% updates
    
    # Initialize arrays and system containers
    X = X0.copy()
    BX = np.zeros_like(X)
    ZX = np.zeros_like(X)
    system.BX_hamiltonian = np.zeros_like(X)
    system.BX_dissipative = np.zeros_like(X)
    system.bx_scalar = np.zeros(1, dtype=np.complex128) 

    # Output parameters
    nsave = nsteps // save_every + 1
    traj = np.empty((nsave, system.M, system.N), dtype=np.complex128)
    time_array = np.empty(nsave, dtype=np.float64)
    traj[0] = X
    time_array[0] = 0.0
    save_idx = 1

    # Arrays to calculate current
    alpha = 0.0 + 0.0j
    traj_current = np.zeros(nsave, dtype=np.complex128)
    traj_current[0] = alpha

    # Time parameters
    t = 0.0
    
    # Other parameters
    safety = 0.9
    kfill = 1.0

    for step in maybe_tqdm(range(1, nsteps + 1), progress_bar, desc="Simulating"):

        # Main integrator step
        z = complex_noise()
        integrator.integrate_step(X, BX, ZX, z, dt, system)
        t += dt


        # Renormalize and calculate ideal time-step every 50/100 steps
        if step % 50 == 0 and step > 0:
            norm = calculate_norm(X)
            X /= norm

        # Calculate current
        if calculate_current:
            alpha = system.euler_step_current(alpha, z, dt, system.bx_scalar, system.kfill , system.k)
        
        # Save data into array
        if step % save_every == 0:
            traj[save_idx] = X
            time_array[save_idx] = t
            if calculate_current:
                traj_current[save_idx] = alpha
            save_idx += 1

            
        ## Print percentage
        #if step % print_every == 0:
        #    print(int(100.0 * step / nsteps), '% Done')

    # Calculate array of timesteps (averged over save_every steps)
    dt_array = np.diff(time_array)/save_every

    return dt_array[:save_idx-1], time_array[:save_idx], traj[:save_idx], traj_current[:save_idx]


def simulate_adaptive_dt(X0, nsteps_max, dt_min, dt_max, tol,
                                     save_every=1, progress_bar=True, calculate_current=False,
                                     integrator=None, system=None):

    if system == None or integrator == None:
        raise Exception("Need to specify integrator and/or system and system.kernel_args()")
    
    print_every = max(1, nsteps_max // 20)   # 5% updates
    
    # Initialize arrays and system containers
    X = X0.copy()
    BX = np.zeros_like(X)
    ZX = np.zeros_like(X)
    system.BX_hamiltonian = np.zeros_like(X)
    system.BX_dissipative = np.zeros_like(X)
    system.bx_scalar = np.zeros(1, dtype=np.complex128) 

    # Output parameters
    nsave = nsteps_max // save_every + 1
    traj = np.empty((nsave, system.M, system.N), dtype=np.complex128)
    time_array = np.empty(nsave, dtype=np.float64)
    traj[0] = X
    time_array[0] = 0.0
    save_idx = 1

    # Arrays to calculate current
    alpha = 0.0 + 0.0j
    traj_current = np.zeros(nsave, dtype=np.complex128)
    traj_current[0] = alpha

    # Time parameters
    t = 0.0
    dt = 1.0*dt_min
    
    # Other parameters
    safety = 0.9
    kfill = 1.0

    for step in maybe_tqdm(range(1, nsteps_max + 1), progress_bar, desc="Simulating"):

        # Main integrator step
        z = complex_noise()
        integrator.integrate_step(X, BX, ZX, z, dt, system)
        t += dt


        # Renormalize and calculate ideal time-step every 50/100 steps
        if step % 50 == 0 and step > 0:
            norm = calculate_norm(X)
            X /= norm
            #system.calculate_drift_matrix(X, BX, system.BX_hamiltonian, system.BX_dissipative, system.bx_scalar, *system.kernel_args())
            dt = choose_dt_from_drift(BX, dt_min, dt_max, tol, safety)

        # Calculate current
        if calculate_current:
            alpha = system.euler_step_current(alpha, z, dt, system.bx_scalar, system.kfill , system.k)
        
        # Save data into array
        if step % save_every == 0:
            traj[save_idx] = X
            time_array[save_idx] = t
            if calculate_current:
                traj_current[save_idx] = alpha
            save_idx += 1

            
        ## Print percentage
        #if step % print_every == 0:
        #    print(int(100.0 * step / nsteps_max), '% Done')

    # Calculate array of timesteps (averged over save_every steps)
    dt_array = np.diff(time_array)/save_every

    return dt_array[:save_idx-1], time_array[:save_idx], traj[:save_idx], traj_current[:save_idx]