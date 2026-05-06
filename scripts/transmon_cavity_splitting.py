import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from openquantum_sde.integrators import EulerMaruyama, splittingRK4EM, splittingRK4Milstein 
from openquantum_sde.integrators import stochasticHeun, splittingExactEuler, splittingExactMidpointEuler, splittingExactIterativeCN
from openquantum_sde.integrators import splittingExactHeun, splittingExactMilstein
from openquantum_sde.systems import TransmonCavity
from openquantum_sde.simulation import simulate_fixed_dt, simulate_adaptive_dt

from openquantum_sde.io import save_trajectory, save_params
from openquantum_sde.utils import calculate_norm, calculate_num_atoms, find_minima_fast
from openquantum_sde.plotting import plot_current, plot_current_phasespace, plot_numatoms_histogram, plot_numatoms_histogram_minimas



#----------------Plotting routines----------------------------------------
    

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


# --------------Main simulation---------------------------

# Transmon/cavity systems parameters and initial conditions
maxAt = 9 #8 #8 #8 #2 #8 #transmon
maxPh = 250 #250 #400 # 400 #10 #400 #photon
k = 1.0 
#Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 
Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 


# Output directories for figs and data
output_figs = True
output_data = True
PROJECT_NAME = "openquantum_sde"
SIM_NAME = "transmon_cavity_test"

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


# Wrapper of simulation to chose the parameters to iterate over
X0 = np.zeros([maxAt+1,maxPh+1], dtype=np.complex128)
X0[0,0] = 1.0 

# Define system
M, N = X0.shape
trans_cavity_system = TransmonCavity(M, N, k, Omega, epsilon, U)

# Define integrator
#myIntegrator = splittingRK4Milstein(M,N)
#myIntegrator = stochasticHeun()
#myIntegrator = splittingExactHeun(taming=True)
#myIntegrator = splittingExactEuler() #taming=True)
#myIntegrator = splittingExactMidpointEuler() #taming=True
myIntegrator = splittingExactIterativeCN()
#myIntegrator = splittingExactMilstein(taming=True)

# Simulation parameters
nsteps = 4000 #10000000 #4000000 #1000000
dt = 5e-4 #5e-5 
save_every = 100
renormalize_every = 1000
time_adaptive = False


if not time_adaptive:
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
else:
    # Run simulation with adaptive dt
    dt_array, times, traj, traj_current = simulate_adaptive_dt(
        X0 = X0, 
        nsteps_max = 150000,
        dt_min = 5e-5, #2e-4,
        dt_max = 2e-4, 
        tol = 0.8,
        save_every = 100, 
        renormalize_every = 100,
        progress_bar=True,
        calculate_current = True,
        integrator = myIntegrator,
        system = trans_cavity_system
        )




#--------------------Output----------------------

# Define parameter dictionary for storge
params = {
    "simulation": {
        "simulation_name": SIM_NAME,
        "dt": dt,
        "nsteps" : nsteps,
        "final_time" : times[-1],
        "save_every" : save_every,
        "renormalize_every" : renormalize_every
    },
    "system": {
        "system_name" : trans_cavity_system.__class__.__name__ ,
        "M": maxAt + 1,
        "N": maxPh + 1,
        "k": k,
        "Omega" : Omega,
        "epsilon" : epsilon,
        "U" :  U
    },
    "numerics": {
        "method": myIntegrator.__class__.__name__ ,
        "time_adaptive" : time_adaptive
    }
}


simidstr = "CK_01"

# Save data
if output_data:
    fname = 'traj_' + simidstr #f"{i:04d}"
    save_trajectory(fname, output_data_dir, times, traj, traj_current, simidstr)
    save_params('params.json', output_data_dir, params)

# Plot figures
if output_figs:
    plot_figures(output_figs_dir, dt, times, traj, traj_current, simidstr)






