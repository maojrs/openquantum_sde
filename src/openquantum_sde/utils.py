import numpy as np
from numba import njit


'''Utility functions for matrix operations and noise calculations.'''

@njit
def complex_noise():
    '''Returns complex noise value'''
    x = np.random.normal()
    y = np.random.normal()
    z = complex(x,y)/np.sqrt(2)
    return z


@njit
def complex_noise_matrix(Z):
    '''Fills matrix Z with complex-valued standard Wiener noise'''
    M, N = Z.shape
    for m in range(M):
        for n in range(N):
            x = np.random.normal()
            y = np.random.normal()
            Z[m, n] = complex(x,y)/np.sqrt(2)


@njit(fastmath = True)
def diag_matrix_exponential(M,N,D, expD):
    '''Calculates matrix expoenntial expD of a diagonal matrix D '''
    for m in range(M):
        for n in range(N):
            expD[m,n] = np.exp(D[m,n])


@njit(fastmath = True)
def calculate_norm(X):
    '''Calculates and returns norm of matrix X'''
    norm = 0.0
    M, N = X.shape
    for m in range(M):
        for n in range(N):
            g = X[m,n]
            norm += g.real * g.real + g.imag * g.imag
    norm = np.sqrt(norm)
    return norm


@njit(fastmath = True)
def calculate_num_atoms(X):
    '''Calculates number of atoms histogram from state matrix X.
    X.shape = (M, N) with M = transmon level (num atoms), N=photon level'''
    M, N = X.shape
    num_atoms = np.zeros(M, dtype=np.float64)
    norm = 0.0
    for m in range(M):
        norm_sqr = 0.0
        for n in range(N):
            g = X[m,n]
            norm_sqr += g.real * g.real + g.imag * g.imag
            norm += g.real * g.real + g.imag * g.imag
        num_atoms[m] = norm_sqr
    return num_atoms/norm


@njit(fastmath = True)
def calculate_num_photons(X):
    '''Calculates expected value of number of photons from state matrix X.
    X.shape = (M, N) with M = transmon level (num atoms), N=photon level'''
    M, N = X.shape
    num_photons = 0.0
    norm = 0.0
    for m in range(M):
        for n in range(N):
            xmn = X[m,n]
            num_photons += n * xmn * (xmn.real - 1.0j*xmn.imag)
            norm += xmn.real * xmn.real + xmn.imag * xmn.imag
    return num_photons/norm


njit(fastmath = True)
def filter_trajectory(traj, traj_current, minima, tolerance):
    '''Filter trajectory to only include data around a minima in the current phase 
    space with a given tolderance radius'''

    #Build boolean mask from traj_current
    mask = np.abs(traj_current - minima) <= tolerance

    # Apply mask along the first axis
    filtered_traj = traj[mask]

    return filtered_traj


def find_minima_fast(traj_current, bins=5, threshold_ratio=0.005):
    '''Fast estimation of minima using 2D histogram peak detection. Not super 
    accurate. More complex algorithms like DBSCAN can be implemented outside 
    of the library.
    
    traj_current : 1D complex trajectory
    bins : Number of bins per dimension
    threshold_ratio : Keep bins with counts >= max_count * threshold_ratio
    
    Returns
    minimas : array of complex numbers
    '''
    
    # Convert to 2D
    x = traj_current.real
    y = traj_current.imag
    
    # 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    
    # Find high-density bins
    max_count = H.max()
    mask = H >= (threshold_ratio * max_count)
    
    # Get bin centers
    centers = []
    for i, j in zip(*np.where(mask)):
        cx = 0.5 * (xedges[i] + xedges[i+1])
        cy = 0.5 * (yedges[j] + yedges[j+1])
        centers.append(cx + 1j * cy)
    
    minimas = np.array(centers)
    print(minimas)
    
    return minimas