import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from openquantum_sde.integrators import EulerMaruyama, splittingRK4EM, splittingRK4Milstein 
from openquantum_sde.integrators import stochasticHeun, splittingExactEuler, splittingExactMidpointEuler, splittingExactIterativeCN
from openquantum_sde.integrators import splittingExactHeun, splittingExactMilstein
from openquantum_sde.systems import TransmonCavity
from openquantum_sde.simulation import simulate_fixed_dt, simulate_adaptive_dt
from openquantum_sde.utils import calculate_norm, calculate_num_atoms
from openquantum_sde.plotting import plot_current, plot_current_phasespace, plot_numatoms_histogram, plot_numatoms_histogram_minimas



#----------------Plotting routines----------------------------------------
    

def plot_figures(dt, times, traj, traj_current, simid):
    if not isinstance(simid, str):
        simid = str(int(simid))
    dt_string = f"{dt:.3g}"

    title1 = 'dt=' + dt_string
    fname1 = "current_timeseries_" + simid + ".png"
    plot_current(times, traj_current, output_dir, fname1, title = title1, savefig = True)

    title2 = 'dt=' + dt_string
    fname2 = "phase_space_trajectory_" + simid + ".png"
    plot_current_phasespace(traj_current, output_dir, fname2, maxval = abs(epsilon)/k, title = title2, savefig = True)

    title3 = 'dt=' + dt_string
    fname3 = "numatoms_histogram_" + simid + ".png"
    plot_numatoms_histogram(traj, output_dir, fname3, title = title3, savefig = True)

    title4 = 'dt=' + dt_string
    minimas = [0.0 + 0.0j, 2.15 + 4.6j, 9.85+ 4.6j]
    fname4 = "numatoms_histogram_minimas" + simid + ".png"
    plot_numatoms_histogram_minimas(traj, traj_current, minimas, output_dir, fname4,  title = title4, savefig = True)


# Transmon/cavity systems parameters and initial conditions
maxAt = 10 #8 #8 #8 #2 #8 #transmon
maxPh = 250 #250 #400 # 400 #10 #400 #photon
k = 1.0 
#Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 
Omega, epsilon, U = 50.0*k, 12.0*k, 400.0*k 



output_dir = Path("figs_test")
output_dir.mkdir(parents=True, exist_ok=True)


# Wrapper of simulation to chose the parameters to iterate over
X0 = np.zeros([maxAt+1,maxPh+1], dtype=np.complex128)
X0[0,0] = 1.0 

# Define system
M, N = X0.shape
trans_cavity_system = TransmonCavity(M, N, k, Omega, epsilon, U)

# Define integrator
dt = 5e-4 #5e-5 #1e-4 #5e-5 #5e-5 #1e-4 #2e-5 #2e-4 #5e-5#2e-4 #8e-5 #8e-5 #4e-4, 3e-4
#myIntegrator = splittingRK4Milstein(M,N)
#myIntegrator = stochasticHeun()
#myIntegrator = splittingExactHeun(taming=True)
#myIntegrator = splittingExactEuler() #taming=True)
#myIntegrator = splittingExactMidpointEuler() #taming=True
myIntegrator = splittingExactIterativeCN()
#myIntegrator = splittingExactMilstein(taming=True)


# Run simulation with fixed dt
dt_array, times, traj, traj_current = simulate_fixed_dt(
    X0 = X0, 
    nsteps = 400000, #10000000, #4000000, #1000000,
    dt = dt, 
    save_every = 100, 
    renormalize_every = 100, #1000,
    progress_bar=True,
    calculate_current = True,
    integrator = myIntegrator,
    system = trans_cavity_system
    )


# Run simulation with adaptive dt
'''dt_array, times, traj, traj_current = simulate_adaptive_dt(
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
    )'''

# Plot figures
plot_figures(dt, times, traj, traj_current, 'test')






