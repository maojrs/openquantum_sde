import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from openquantum_sde.integrators import EulerMaruyama, splittingEMRK4
from openquantum_sde.systems import TransmonCavity
from openquantum_sde.simulation import simulate_fixed_dt, simulate_adaptive_dt
from openquantum_sde.utils import calculate_norm, calculate_num_atoms

# Transmon/cavity systems parameters and initial conditions
maxAt = 8 #8 #8 #2 #8 #transmon
maxPh = 250 #250 #400 # 400 #10 #400 #photon
k = 1.0 
Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 

dt_base = 2e-4
nsteps_base = 5000000
dt = dt_base
nsteps = nsteps_base

output_dir = Path("figs")
output_dir.mkdir(parents=True, exist_ok=True)


def plot_figures(dt, times, traj, traj_current):
    dt_percent = str(int(dt_base/dt))
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
    fname2 = "current_timeseries_" + dt_percent + ".png"
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
    fname2 = "phase_sapce_trajectory_" + dt_percent + ".png"
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
    fname3 = "numatoms_histogram_" + dt_percent + ".png"
    fig3.savefig(output_dir / fname3)




# Wrapper of simulation to chose the parameters to iterate over
def parallel_simulation_wrapper(dt, nsteps):
    X0 = np.zeros([maxAt+1,maxPh+1], dtype=np.complex128)
    X0[0,0] = 1.0 

    # Define integrator
    myIntegrator = splittingEMRK4()

    # Define system
    M, N = X0.shape
    trans_cavity_system = TransmonCavity(M, N, k, Omega, epsilon, U)

    # Run simulation
    dt_array, times, traj, traj_current = simulate_fixed_dt(
        X0 = X0, 
        nsteps = nsteps,
        dt = dt, 
        save_every = 100, 
        progress_bar=False,
        calculate_current = True,
        integrator = myIntegrator,
        system = trans_cavity_system
        )
    
    # Process data in files perhaps (e.g. how many minima were found)
    #print("dt=", dt, " Done")
    plot_figures(dt, times, traj, traj_current)


# An additional wrapper that takes as input the variable parameters and returns the simulation
def run_simulation(params):
    return parallel_simulation_wrapper(
        dt=params["dt"],
        nsteps=params["nsteps"]
        )

 # Create paremeter list
param_list = []
for i in range(25):
    param_list.append({
        "dt": dt,
        "nsteps": nsteps
    })
    dt = 0.8 * dt 
    nsteps = int(1.2*nsteps)



def run_all(param_list, use_progress=True):
    with ProcessPoolExecutor() as executor:
        iterator = executor.map(run_simulation, param_list)
        
        if use_progress:
            iterator = tqdm(iterator, total=len(param_list), desc="Simulations")
        
        return list(iterator)

run_all(param_list)