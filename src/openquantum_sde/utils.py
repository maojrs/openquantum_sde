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