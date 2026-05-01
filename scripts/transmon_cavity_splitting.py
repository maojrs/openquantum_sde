import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from openquantum_sde.integrators import EulerMaruyama, splittingRK4EM, splittingRK4Milstein 
from openquantum_sde.integrators import stochasticHeun, splittingExactEuler, splittingExactSemiEuler, splittingExactHeun, splittingExactMilstein
from openquantum_sde.systems import TransmonCavity
from openquantum_sde.simulation import simulate_fixed_dt, simulate_adaptive_dt
from openquantum_sde.utils import calculate_norm, calculate_num_atoms


#----------------Plotting routines----------------------------------------
    

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
    scale = 0.8 #1.0
    maxval = scale*(abs(epsilon)/k)
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=120)
    ax2.plot(traj_current.real, traj_current.imag, lw=0.2, color='k')
    ax2.set_title('dt=' + dt_string) 
    ax2.set_xlim([-maxval,maxval])
    ax2.set_ylim([-maxval,maxval])
    ax2.set_aspect('equal')
    fname2 = "phase_space_trajectory_" + simid + ".png"
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


# Transmon/cavity systems parameters and initial conditions
maxAt = 8 #8 #8 #2 #8 #transmon
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
dt = 2e-4 #5e-5#2e-4 #8e-5 #8e-5 #4e-4, 3e-4
#myIntegrator = splittingRK4Milstein(M,N)
#myIntegrator = stochasticHeun()
#myIntegrator = splittingExactHeun(taming=True)
#myIntegrator = splittingExactEuler(taming=True)
myIntegrator = splittingExactSemiEuler() #taming=True)
#myIntegrator = splittingExactMilstein(taming=True)



# Run simulation with fixed dt
dt_array, times, traj, traj_current = simulate_fixed_dt(
    X0 = X0, 
    nsteps = 1000000, #4000000, #1000000,
    dt = dt, 
    save_every = 100, 
    renormalize_every = 100,
    progress_bar=True,
    calculate_current = True,
    integrator = myIntegrator,
    system = trans_cavity_system
    )

# Run simulation with adaptive dt
'''dt_array, times, traj, traj_current = simulate_adaptive_dt(
    X0 = X0, 
    nsteps_max = 150000,
    dt_min = 8e-5, #2e-4,
    dt_max = 1e-4, 
    tol = 0.8,
    save_every = 100, 
    renormalize_every = 100,
    progress_bar=True,
    calculate_current = True,
    integrator = myIntegrator,
    system = trans_cavity_system
    )'''

# Plot figures
plot_figures(dt, times, traj, traj_current, 1)






