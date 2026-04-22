from numba import njit
import numpy as np


'''Utilities fucntions'''

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
    '''Calculates number of atoms histogram fro state matrix X. It 
    assumes X(m,n) is a quantum state matrix, where m corresponds to the
    number of atoms state and n to the number of photons state. '''
    M, N = X.shape
    num_atoms = np.zeros(M, dtype=np.float64)
    for m in range(M):
        norm_sqr = 0.0
        for n in range(N):
            g = X[m,n]
            norm_sqr += g.real * g.real + g.imag * g.imag
        num_atoms[m] = norm_sqr
    return num_atoms