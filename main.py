import numpy as np
import time

from openquantum_sde.simulation import simulate_em_adaptive
from openquantum_sde.systems.transmon_cavity import  TransmonCavity

maxAt_1 = 0
maxPh_1 = 50 
k, Omega, epsilon, U = [1.0, 0.0, 5.0, 0.0]
X0 = np.zeros([maxAt_1+1,maxPh_1+1], dtype=np.complex128)
X0[0,0] = 1.0

# Define system
M, N = X0.shape
trans_cavity_system = TransmonCavity(M, N, k, Omega, epsilon, U)


start = time.perf_counter()
dt_array_1, times_1, traj_1, traj_current_1 = simulate_em_adaptive(
    X0 = X0, 
    nsteps_max = 3000, #300000, 
    dt_min = 1e-6, #0.0005/10.0,
    dt_max = 0.05, #0.0005*10.0,
    tol = 0.5,
    save_every = 1,
    calculate_current = True,
    system = trans_cavity_system)
end = time.perf_counter()
print(f"Time taken: {end - start:.6f} seconds")