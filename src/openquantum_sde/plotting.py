import numpy as np
import math
import matplotlib.pyplot as plt

from openquantum_sde.utils import calculate_num_atoms, filter_trajectory


def plot_current(times, traj_current, output_dir = None, fname = None, title = None, savefig = False):
    '''Plots current (real, imag and squared norm) as a fucntion of time'''
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(times, traj_current.real, label=r'Re[$\alpha$]', lw=0.5)
    ax.plot(times, traj_current.imag, label=r'Im[$\alpha$]', lw=0.5)
    ax.plot(times, (traj_current*traj_current.conjugate()).real, label=r'$|\alpha^2|$', lw=0.5)
    if title != None:
        ax.set_title(title) 
    ax.set_xlabel('Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if savefig:
        fig.savefig(output_dir / fname)


def plot_current_phasespace(traj_current, output_dir = None, fname = None, maxval = 10, scale = 1.0, title = None, savefig = False):
    '''Plots phase space trajectory of current, real vs imaginary part'''
    maxval = maxval * scale
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.plot(traj_current.real, traj_current.imag, lw=0.2, color='k')
    if title != None:
        ax.set_title(title) 
    ax.set_xlim([-maxval,maxval])
    ax.set_ylim([-maxval,maxval])
    ax.set_aspect('equal')
    if savefig:
        fig.savefig(output_dir / fname)


def plot_numatoms_histogram(traj, output_dir = None, fname = None,  title = None, savefig = False):
    '''Plots num_atoms histogram (averaged over whole given time interval)'''

    # Arrays to calculate num atoms
    traj_num_atoms = np.zeros(traj.shape[0:2], dtype=np.float64)
    for i in range(traj.shape[0]):
        traj_num_atoms[i] = calculate_num_atoms(traj[i])

    imin = 0
    imax = len(traj_num_atoms)
    mean_num_atoms = np.mean(traj_num_atoms[imin:imax], axis=0)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.bar(np.arange(0, len(mean_num_atoms)), mean_num_atoms)
    if title != None:
        ax.set_title(title)  
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    if savefig:
        fig.savefig(output_dir / fname)


def plot_numatoms_histogram_minimas(traj, traj_current, minimas, output_dir = None, fname = None,  title = None, savefig = False):
    '''Plots the number of atoms histograms for different minimas given in the current phase space'''
    
    numplots = len(minimas)

    # Determine layout
    if numplots <= 3:
        rows, cols = 1, numplots
    elif numplots == 4:
        rows, cols = 2, 2
    else:
        cols = 3
        rows = math.ceil(numplots / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    # Ensure ax is iterable (flatten in case of multiple rows/cols)
    if numplots == 1:
        ax = [ax]
    else:
        ax = np.array(ax).flatten()

    # Plot on each axis
    for j, axis in enumerate(ax):
        if j < numplots:

            # Filter trajectory around given minima
            filtered_traj = filter_trajectory(traj, traj_current, minimas[j], tolerance = 1.0)

            if filtered_traj.size > 0:
                # Array to calculate num atoms
                traj_num_atoms = np.zeros(filtered_traj.shape[0:2], dtype=np.float64)
                for i in range(filtered_traj.shape[0]):
                    traj_num_atoms[i] = calculate_num_atoms(filtered_traj[i])

                imin = 0
                imax = len(traj_num_atoms)
                mean_num_atoms = np.mean(traj_num_atoms[imin:imax], axis=0)
                axis.bar(np.arange(0, len(mean_num_atoms)), mean_num_atoms)
                if title != None:
                    axis.set_title(title)  
                axis.set_xlabel("Value")
                axis.set_ylabel("Frequency")
            else:
                axis.set_visible(False) # Hide empty subplots
        else:
            axis.set_visible(False)  # Hide unused subplots
    
    if savefig:
        fig.savefig(output_dir / fname)

