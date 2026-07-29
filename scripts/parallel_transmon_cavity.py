import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import h5py
import gc
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
numsims = 10
total_cores = os.cpu_count()
workers = 5 #max(1, total_cores - 2)

# For progress bar
tqdm.set_lock(RLock())


# Transmon/cavity systems parameters
#NEEEED TO SCALE THIS GUY UP TO 20 AND RUN A TEST
maxAt = 20 #13 #11 #9 #8 #8 #8 #2 #8 #transmon
maxPh = 320 #300 #250 #250 #400 # 400 #10 #400 #photon
k = 1.0 
#Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 
Omega, epsilon, U = 50.0*k, 10.0*k, 400.0*k

# Simulation parameters
nsteps = 25000000 #10000000 #4000000 #1000000
dt = 2.0e-4 #2.5e-4 #2.5e-4 #5e-5 
save_every = 200 #100
renormalize_every = 1 #1000
time_adaptive = False

# Aliases for integrator and system classes
thisSystem = TransmonCavity
thisIntegrator = splittingExactIterativeCN


# Output directories for figs and data
output_figs = True
output_data = True
PROJECT_NAME = "openquantum_sde"
SIM_NAME = "transmon_cavity_eps_" + str(int(epsilon))

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


# Minimas in phase space depending on drive epsilon (for phase space plots)
minimas_by_epsilon = {
    11: [0.00 + 0.00j, 2.18 + 4.39j, 9.79 + 3.45j],
    12: [0.00 + 0.00j, 2.15 + 4.60j, 9.84 + 4.61j],
    13: [0.00 + 0.01j, 2.14 + 4.82j, 9.85 + 5.57j],
    14: [0.00 + 0.01j, 2.14 + 5.04j, 9.86 + 6.39j],
    15: [0.00 + 0.01j, 2.15 + 5.25j, 9.85 + 7.12j],
    16: [0.00 + 0.01j, 2.17 + 5.47j, 9.86 + 7.78j],
    17: [0.00 + 0.01j, 2.19 + 5.70j, 9.89 + 8.39j],
    18: [0.00 + 0.01j, 2.23 + 5.93j, 9.93 + 8.95j],
    19: [0.00 + 0.01j, 2.27 + 6.16j, 9.98 + 9.49j],
    20: [0.00 + 0.01j, 2.32 + 6.40j, 10.04 + 10.00j],
}

minimas = minimas_by_epsilon.get(int(epsilon/k), [])


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
        progress_bar=False,
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
        plot_figures(output_figs_dir, dt, times, traj, traj_current, minimas, simidstr)

    # Explicit cleanup
    del traj
    del traj_current
    del times
    del dt_array
    del trans_cavity_system
    del myIntegrator
    del X0

    plt.close('all')

    gc.collect()


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
    

def plot_figures(output_dir, dt, times, traj, traj_current, minimas, simid):
    if not isinstance(simid, str):
        simid = str(int(simid))
    dt_string = f"{dt:.3g}"

    #title1 = 'dt=' + dt_string
    title1 = r"Current $\alpha$"
    fname1 = "current_timeseries_" + simid + ".png"
    plot_current(times, traj_current, output_dir, fname1, title = title1, savefig = True)

    #title2 = 'dt=' + dt_string
    title2 = "Phase space"
    fname2 = "phase_space_trajectory_" + simid + ".png"
    lim = abs(epsilon)/k
    plot_current_phasespace(traj_current, output_dir, fname2, xlim = [-3.0, 14], ylim = [-3.0, 14], minimas = minimas, title = title2, savefig = True)

    #title3 = 'dt=' + dt_string
    #fname3 = "numatoms_histogram_" + simid + ".png"
    #plot_numatoms_histogram(traj, output_dir, fname3, title = title3, savefig = True)

    #title4 = 'dt=' + dt_string
    title4 = ''
    fname4 = "histograms_natoms_minimas_" + simid + ".png"
    plot_numatoms_histogram_minimas(traj, traj_current, minimas, output_dir, fname4,  title = title4, savefig = True)



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

# Save parameters file
if output_data:
    save_params('params.json', output_data_dir, params)
if output_figs:
    save_params('params.json', output_figs_dir, params)




def main():
    # Create paremeter list for parallel runs (just sim ids)
    param_list = []
    for i in range(numsims):
        param_list.append({
            "simid": i+1,
        })
        
    # Run parallelized simulation
    run_all(param_list)


if __name__ == "__main__":
    main()
