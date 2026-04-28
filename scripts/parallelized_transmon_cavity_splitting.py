import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
from pathlib import Path
from multiprocessing import RLock
from concurrent.futures import ProcessPoolExecutor

from openquantum_sde.integrators import EulerMaruyama, splittingRK4EM
from openquantum_sde.systems import TransmonCavity
from openquantum_sde.simulation import simulate_fixed_dt, simulate_adaptive_dt
from openquantum_sde.utils import calculate_norm, calculate_num_atoms

# For parallelizations
numsims = 15
total_cores = os.cpu_count()
workers = max(1, total_cores - 2)
dt_base = 2e-4
nsteps_base = 500
save_every_base = 100
dt = dt_base
nsteps = nsteps_base
save_every = save_every_base
finaltime = dt * nsteps
totalframes = int(nsteps/save_every)

# For progress bar
tqdm.set_lock(RLock())

# Transmon/cavity systems parameters and initial conditions
maxAt = 8 #8 #8 #2 #8 #transmon
maxPh = 250 #250 #400 # 400 #10 #400 #photon
k = 1.0 
Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 



output_dir = Path("figs")
output_dir.mkdir(parents=True, exist_ok=True)


# Wrapper of simulation to chose the parameters to iterate over
def parallel_simulation_wrapper(dt, nsteps, save_every, barposition):
    X0 = np.zeros([maxAt+1,maxPh+1], dtype=np.complex128)
    X0[0,0] = 1.0 

    # Define integrator
    myIntegrator = splittingRK4EM()

    # Define system
    M, N = X0.shape
    trans_cavity_system = TransmonCavity(M, N, k, Omega, epsilon, U)

    # Parameters for parallelized progress bar
    tqdm_kwargs = {
        "position": (barposition - 1)%workers + 1,
        "leave": False,
        "desc": f"Sim {barposition}",
        "dynamic_ncols": True,
        "ascii": True}

    # Run simulation
    dt_array, times, traj, traj_current = simulate_fixed_dt(
        X0 = X0, 
        nsteps = nsteps,
        dt = dt, 
        save_every = save_every, 
        renormalize_every = 10,
        progress_bar=True,
        tqdm_kwargs=tqdm_kwargs,
        calculate_current = True,
        integrator = myIntegrator,
        system = trans_cavity_system
        )
    
    # Process data in files perhaps (e.g. how many minima were found)
    #print("dt=", dt, " Done")
    plot_figures(dt, times, traj, traj_current, barposition)


# An additional wrapper that takes as input the variable parameters and returns the simulation
def run_simulation(params):
    return parallel_simulation_wrapper(
        dt=params["dt"],
        nsteps=params["nsteps"],
        save_every=params["save_every"],
        barposition=params["barposition"]
        )


def run_all(param_list, use_progress=True):
    

    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(run_simulation, param_list)

        if use_progress:
            iterator = tqdm(iterator, total=len(param_list), desc="All simulations", position=0, leave=True)

        return list(iterator)
    

def plot_figures(dt, times, traj, traj_current, barposition):
    simid = str(int(barposition))
    dt_string = f"{dt:.3g}"

    # plot current
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(times, traj_current.real, label=r'Re[$\alpha$]', lw=0.5)
    ax.plot(times, traj_current.imag, label=r'Im[$\alpha$]', lw=0.5)
    ax.plot(times, (traj_current*traj_current.conjugate()).real, label=r'$|\alpha^2|$', lw=0.5)
    ax.set_title('dt=' + dt_string) 
    ax.set_xlabel('Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname2 = "current_timeseries_" + simid + ".png"
    fig.savefig(output_dir / fname2)

    # Plot phase space of current
    scale = 1.0
    maxval = scale*(abs(epsilon)/k)
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=120)
    ax2.plot(traj_current.real, traj_current.imag, lw=0.2, color='k')
    ax2.set_title('dt=' + dt_string) 
    ax2.set_xlim([-maxval,maxval])
    ax2.set_ylim([-maxval,maxval])
    ax2.set_aspect('equal')
    fname2 = "phase_sapce_trajectory_" + simid + ".png"
    fig2.savefig(output_dir / fname2)

    # Number of atoms histogram
    # Arrays to calculate num atoms
    traj_num_atoms = np.zeros(traj.shape[0:2], dtype=np.float64)
    for i in range(traj.shape[0]):
        traj_num_atoms[i] = calculate_num_atoms(traj[i])
    
    # Plot num_atoms histogram (averaged over time interval)
    imin = 0
    imax = len(traj_num_atoms)
    mean_num_atoms = np.mean(traj_num_atoms[imin:imax], axis=0)
    fig3, ax3 = plt.subplots(figsize=(6, 6), dpi=120)
    ax3.bar(np.arange(0, len(mean_num_atoms)), mean_num_atoms)
    ax3.set_title('dt=' + dt_string) 
    ax3.set_xlabel("Value")
    ax3.set_ylabel("Frequency")
    fname3 = "numatoms_histogram_" + simid + ".png"
    fig3.savefig(output_dir / fname3)



 # Create paremeter list
param_list = []
for i in range(numsims):
    param_list.append({
        "dt": dt,
        "nsteps": nsteps,
        "save_every": save_every,
        "barposition": i + 1  # for progress bar
    })
    dt = 0.8 * dt 
    nsteps = int(finaltime/dt)
    save_every = int(nsteps/totalframes)


# Run parallelized simulation
run_all(param_list)