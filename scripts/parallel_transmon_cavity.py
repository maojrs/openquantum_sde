import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # For parallel writing differen files
import h5py
from pathlib import Path
from multiprocessing import RLock
from concurrent.futures import ProcessPoolExecutor

from openquantum_sde.integrators import EulerMaruyama, splittingRK4EM, splittingRK4Milstein 
from openquantum_sde.integrators import stochasticHeun, splittingExactEuler, splittingExactMidpointEuler, splittingExactIterativeCN
from openquantum_sde.integrators import splittingExactHeun, splittingExactMilstein
from openquantum_sde.systems import TransmonCavity
from openquantum_sde.simulation import simulate_fixed_dt, simulate_adaptive_dt

from openquantum_sde.io import save_trajectory, save_params
from openquantum_sde.utils import calculate_norm, calculate_num_atoms, find_minima_fast
from openquantum_sde.plotting import plot_current, plot_current_phasespace, plot_numatoms_histogram, plot_numatoms_histogram_minimas

# For parallelizations
numsims = 15
total_cores = os.cpu_count()
workers = max(1, total_cores - 2)

# For progress bar
tqdm.set_lock(RLock())


# Output directories for figs and data
output_figs = True
output_data = True
PROJECT_NAME = "openquantum_sde"
SIM_NAME = "transmon_cavity_CN_parallel"

if "DATA" in os.environ:
    base_dir = Path(os.environ["DATA"]).expanduser()
else:
    base_dir = Path(".")  # current folder

simulation_dir = base_dir / PROJECT_NAME / SIM_NAME

if output_figs:
    output_figs_dir = simulation_dir / "figs"
    output_figs_dir.mkdir(parents=True, exist_ok=True)

if output_data:
    output_data_dir = simulation_dir / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)


# Transmon/cavity systems parameters
maxAt = 9 #8 #8 #8 #2 #8 #transmon
maxPh = 250 #250 #400 # 400 #10 #400 #photon
k = 1.0 
#Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 
Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 

# Simulation parameters
nsteps = 4000 #10000000 #4000000 #1000000
dt = 5e-4 #5e-5 
save_every = 100
renormalize_every = 1000
time_adaptive = False

# Aliases for inetgartor and system classes
thisSystem = TransmonCavity
thisIntegrator = splittingExactIterativeCN


# Wrapper of simulation 
def parallel_simulation_wrapper(simid):
    # Wrapper of simulation to chose the parameters to iterate over
    X0 = np.zeros([maxAt+1,maxPh+1], dtype=np.complex128)
    X0[0,0] = 1.0 

    # Define system
    M, N = X0.shape
    trans_cavity_system = thisSystem(M, N, k, Omega, epsilon, U)

    # Define integrator
    myIntegrator = thisIntegrator()

    # Parameters for parallelized progress bar
    tqdm_kwargs = {
        "position": (simid - 1)%workers + 1,
        "leave": False,
        "desc": f"Sim {simid}",
        "dynamic_ncols": True,
        "ascii": True}

    # Run simulation with fixed dt
    dt_array, times, traj, traj_current = simulate_fixed_dt(
        X0 = X0, 
        nsteps = nsteps, 
        dt = dt, 
        save_every = save_every, 
        renormalize_every = renormalize_every,                   
        progress_bar=True,
        calculate_current = True,
        integrator = myIntegrator,
        system = trans_cavity_system
        )

    simidstr = "CK_" + f"{simid:04d}"

    # Save data
    if output_data:
        fname = 'traj_' + simidstr
        save_trajectory(fname, output_data_dir, times, traj, traj_current, simidstr)

    # Plot figures
    if output_figs:
        plot_figures(output_figs_dir, dt, times, traj, traj_current, simidstr)


# An additional wrapper that takes as input the parameters and returns the simulation
def run_simulation(params):
    return parallel_simulation_wrapper(
        simid=params["simid"],
        )


def run_all(param_list, use_progress=True):
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(run_simulation, param_list)

        if use_progress:
            iterator = tqdm(iterator, total=len(param_list), desc="All simulations", position=0, leave=True)

        return list(iterator)



#----------------Plotting routine----------------------------------------
    

def plot_figures(output_dir, dt, times, traj, traj_current, simid):
    if not isinstance(simid, str):
        simid = str(int(simid))
    dt_string = f"{dt:.3g}"

    minimas = [0.0 + 0.0j, 2.15 + 4.6j, 9.85+ 4.6j]


    title1 = 'dt=' + dt_string
    fname1 = "current_timeseries_" + simid + ".png"
    plot_current(times, traj_current, output_dir, fname1, title = title1, savefig = True)

    title2 = 'dt=' + dt_string
    fname2 = "phase_space_trajectory_" + simid + ".png"
    lim = abs(epsilon)/k
    plot_current_phasespace(traj_current, output_dir, fname2, pltlims = [-0.25*lim, lim], minimas = minimas, title = title2, savefig = True)

    #title3 = 'dt=' + dt_string
    #fname3 = "numatoms_histogram_" + simid + ".png"
    #plot_numatoms_histogram(traj, output_dir, fname3, title = title3, savefig = True)

    title4 = 'dt=' + dt_string
    fname4 = "histograms_natoms_minimas_" + simid + ".png"
    plot_numatoms_histogram_minimas(traj, traj_current, minimas, output_dir, fname4,  title = title4, savefig = True)



# Create paremeter list (just sim ids)
param_list = []
for i in range(numsims):
    param_list.append({
        "simid": i+1,
    })
    
# Run parallelized simulation
run_all(param_list)


# Define parameter dictionary for storge and otput parameters
params = {
    "simulation": {
        "simulation_name": SIM_NAME,
        "dt": dt,
        "nsteps" : nsteps,
        "final_time" : dt * nsteps,
        "save_every" : save_every,
        "renormalize_every" : renormalize_every
    },
    "system": {
        "system_name" : thisSystem.__name__ ,
        "M": maxAt + 1,
        "N": maxPh + 1,
        "k": k,
        "Omega" : Omega,
        "epsilon" : epsilon,
        "U" :  U
    },
    "numerics": {
        "method": thisIntegrator.__name__ ,
        "time_adaptive" : time_adaptive
    }
}

# Save data
if output_data:
    save_params('params.json', output_data_dir, params)