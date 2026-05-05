import numpy as np
from numba import njit

from openquantum_sde.systems.system import base_system


'''Main systems class with system specific parmeters + numba source routines'''

class TransmonCavity(base_system):
    '''Coupled transmon-cavity system: routines to obtain the 
    Stochastic Schrödinger equation (SSE).

    This module defines the drift and noise terms for the SSE of 
    a coupled transmon-cavity system by defining the corresponding
    drift and noise matrices.

    State of the sytem given by:
    X: Probability amplitude (Matrix of C coefficients)
    X.shape = (M, N) with M = transmon level (num atoms), N=photon level

    Main functions (note code is not re-used so the remain static under numba)
    ---------
    compute_exponential_drift_matrix_diagonal: Calculates exponential matrix of 
    diagonal part of the coherent part of the drift matrix

    drift_matrix_coherent: computes the coherent part of the drift (Hamiltonian
    + diagonal of dissipative part (loss)).

    drift_matrix_noncoherent: computes the noncoherent part of the drift (
    non-diagonal part of the dissipative drift).

    calculate_drift_matrix_nondiagonal: calculates total drift without the diagonal
    coherent part (to use together with compute_exponential_drift_matrix_diagonal)

    calculate_drift_matrix: calculates total drift (coherent + non-coherent)

    calculate_noise_matrix: computes the noise (dissipative)
    '''

    def __init__(self, M, N, k, Omega, epsilon, U):
        self.M = M
        self.N = N
        self.k = k
        self.Omega = Omega
        self.epsilon = epsilon
        self.U = U
        self.kfill = 1.0 * k

        '''# IMPLEMENTED INTO THE PRECOMPUTATIONS OF EACH INTEGRATOR
        # Define auxiliary containers used by integrators 
        # (add more if needed, to avoid defining arrays at integration steps)
        self.expdiagBX = np.zeros([M,N], dtype=np.complex128)
        self.BXtmp = np.zeros([M,N], dtype=np.complex128)
        self.BX_coherent = np.zeros([M,N], dtype=np.complex128)
        self.BX_noncoherent = np.zeros([M,N], dtype=np.complex128)
        self.ZXtmp = np.zeros([M,N], dtype=np.complex128)
        self.bx_scalar = np.zeros(1, dtype=np.complex128)''' 

        # Precompute constant arrays used in the class routines
        self.sqrt_n, self.sqrt_n1, self.sqrt_m_n1, self.sqrt_m1_n, self.sqrt_k_n1 = self.precompute_arrays(self.M, self.N, self.k)


    def parameters(self):
        return self.M, self.N, self.k, self.Omega, self.epsilon, self.U

    def sqrt_arrays(self):
        return self.sqrt_n, self.sqrt_n1, self.sqrt_m_n1, self.sqrt_m1_n, self.sqrt_k_n1
    
    def kernel_args(self):
        return (self.M, self.N, self.k, self.Omega, self.epsilon, self.U,
        self.sqrt_n, self.sqrt_n1, self.sqrt_m_n1, self.sqrt_m1_n, self.sqrt_k_n1)


    @staticmethod
    @njit(fastmath = True)
    def precompute_arrays(M,N,k):
        ''' Precomputes and return all square roots and constant 
        arrays used to calculate the drift and noise matrices.'''
        sqrt_n = np.zeros(N, dtype=np.float64)
        sqrt_n1 = np.zeros(N, dtype=np.float64)
        sqrt_m_n1 = np.zeros((M, N), dtype=np.float64)
        sqrt_m1_n = np.zeros((M, N), dtype=np.float64)
        sqrt_k_n1 = np.zeros(N, dtype=np.float64)

        for m in range(M):
            for n in range(N):
                sqrt_m_n1[m, n] = np.sqrt(m * (n + 1.0))
                if m < M-1:
                    sqrt_m1_n[m, n] = np.sqrt((m + 1.0) * n)
        for n in range(N):
            sqrt_n[n] = np.sqrt(n)
            sqrt_n1[n] = np.sqrt(n + 1.0)
            sqrt_k_n1[n] = np.sqrt(k * (n + 1))
        return sqrt_n, sqrt_n1, sqrt_m_n1, sqrt_m1_n, sqrt_k_n1
    

    @staticmethod
    @njit(fastmath = True)
    def compute_exponential_drift_matrix_diagonal(expdiagBX, dt,
                                M, N, k, Omega, epsilon, U, 
                                sqrt_n, sqrt_n1, sqrt_m_n1, sqrt_m1_n, sqrt_k_n1):
        '''
        Calculates the diagonal part of the drift matrix .
        Takes as input the system parameters and the precalculated
        square root arrays. Only needs to be calculted once for a given dt.

        X: Probability amplitude (Matrix of C coefficients)
        '''
        for m in range(M):
            for n in range(N):
                s = (1j * 0.5 * U * m * (m - 1.0) - k * n) 
                expdiagBX[m,n] = np.exp(dt * s)


    @staticmethod
    @njit(fastmath = True)
    def drift_matrix_coherent(X, BX_coherent, 
                              M, N, k, Omega, epsilon, U, 
                              sqrt_n, sqrt_n1, sqrt_m_n1, sqrt_m1_n, sqrt_k_n1):
        '''
        Calculates the Hamiltonian part (+diagonal dissipations) of the drift matrix .
        Takes as input the system parameters and the precalculated
        square root arrays.

        X: Probability amplitude (Matrix of C coefficients)
        BX_coherent: Resulting drift matrix by multiplying drift tensor by X.
        '''
        for m in range(M):
            for n in range(N):
                s = (1j * 0.5 * U * m * (m - 1.0) - k*n) * X[m, n]
                if m > 0 and n < N-1:
                    s += Omega * sqrt_m_n1[m, n] * X[m - 1, n + 1]
                if m < M-1 and n > 0:
                    s += -Omega * sqrt_m1_n[m, n] * X[m + 1, n - 1]
                if n > 0:
                    s += epsilon * sqrt_n[n] * X[m, n - 1]
                if n < N-1:
                    s -= epsilon * sqrt_n1[n] * X[m, n + 1]
                BX_coherent[m, n] = s


    @staticmethod
    @njit(fastmath = True)
    def drift_matrix_noncoherent(X, BX_noncoherent, bx_scalar,
                                 M, N, k, Omega, epsilon, U, 
                                 sqrt_n, sqrt_n1, sqrt_m_n1, sqrt_m1_n, sqrt_k_n1):
        '''
        Calculates the dissipative part of the drift matrix.
        Takes as input only the k parmeter and a precalculated
        square root array.

        X: Probability amplitude (Matrix of C coefficients)
        BX2: Resulting drift matrix (second part)
        bx2: Resulting drift scalar (just a part of BX2 but needed to calculate current)
        stored as 1D complex np.array to be passed as reference
        '''
        # Calculate the drift scalar bx
        bx = 0.0 + 0.0j
        norm = 0.0
        for m in range(M):
            for n in range(N):
                if n < N-1:
                    bx += X[m,n] * X[m,n+1].conjugate() * sqrt_n1[n]
                norm += (X[m,n] * X[m,n].conjugate()).real
        bx = np.sqrt(2 * k) * bx / norm
        bx_scalar[0] = 1.0 * bx

        # Calculate the drift matrix BX2 (modifies the passed array)
        for m in range(M):
            for n in range(N):
                if n < N - 1:
                    BX_noncoherent[m,n] = np.sqrt(2) * bx * sqrt_k_n1[n] * X[m,n+1]
                else:
                    BX_noncoherent[m,n] = 0.0 + 0.0j


    @staticmethod
    @njit(fastmath = True)
    def calculate_drift_scalar(X, bx_scalar, 
                     M, N, k, Omega, epsilon, U, 
                     sqrt_n, sqrt_n1, sqrt_m_n1, sqrt_m1_n, sqrt_k_n1):
        '''
        Calculates the drift scalar part only.
        Takes as input only the k parmeter and a precalculated
        square root array.

        X: Probability amplitude (Matrix of C coefficients)
        bx2: Resulting drift scalar (just a part of noncoherent drift 
        but also needed to calculate current)
        stored as 1D complex np.array to be passed as reference
        '''
        # Calculate the drift scalar bx only
        bx = 0.0 + 0.0j
        norm = 0.0
        for m in range(M):
            for n in range(N):
                xmn = X[m, n]
                if n < N-1:
                    xmn1 = X[m, n + 1]
                    bx += xmn * (xmn1.real - 1j * xmn1.imag) * sqrt_n1[n]
                norm += xmn.real * xmn.real + xmn.imag * xmn.imag
        bx = np.sqrt(2 * k) * bx / norm
        bx_scalar[0] = 1.0 * bx


    @staticmethod
    @njit(fastmath=True)
    def calculate_drift_matrix_nondiagonal(X, BX, bx_scalar,
                    M, N, k, Omega, epsilon, U, 
                    sqrt_n, sqrt_n1, sqrt_m_n1, sqrt_m1_n, sqrt_k_n1):
        '''Calculates the total drift matrix (drift_matrix_hamiltonian + 
        drift_matrix_dissipative) and stores it on BX. Less readable but 
        optimized for efficiency.'''

        bx = 0.0 + 0.0j
        norm = 0.0

        # Calculate the Hamiltonian drift matrix (modifies the passed array)
        for m in range(M):
            for n in range(N):
                xmn = X[m, n]
                s = 0.0 + 0.0j
                if n < N-1:
                    xmn1 = X[m, n + 1]
                    s -= epsilon * sqrt_n1[n] * xmn1
                    bx += xmn * (xmn1.real - 1j * xmn1.imag) * sqrt_n1[n]
                if n > 0:
                    s += epsilon * sqrt_n[n] * X[m, n - 1]
                if m > 0 and n < N-1:
                    s += Omega * sqrt_m_n1[m, n] * X[m - 1, n + 1]
                if m < M-1 and n > 0:
                    s -= Omega * sqrt_m1_n[m, n] * X[m + 1, n - 1]
                BX[m, n] = s

                 # Calculate the drift scalar bx
                norm += xmn.real * xmn.real + xmn.imag * xmn.imag
        bx = np.sqrt(2 * k) * bx / norm
        bx_scalar[0] = 1.0 * bx

        # Calculate the dissipative drift matrix and the total drift (modifies the passed array)
        for m in range(M):
            for n in range(N):
                if n < N - 1:
                    s2 = np.sqrt(2) * bx * sqrt_k_n1[n] * X[m,n+1]
                else:
                    s2 = 0.0 + 0.0j
                
                # Calculates total drift matrix
                BX[m, n] += s2


    @staticmethod
    @njit(fastmath=True)
    def calculate_drift_matrix(X, BX, BX_coherent, BX_noncoherent, bx_scalar,
                    M, N, k, Omega, epsilon, U, 
                    sqrt_n, sqrt_n1, sqrt_m_n1, sqrt_m1_n, sqrt_k_n1):
        '''Calculates the total drift matrix (drift_matrix_hamiltonian + 
        drift_matrix_dissipative) and stores it on BX. Less readable but 
        optimized for efficiency.'''

        bx = 0.0 + 0.0j
        norm = 0.0

        # Calculate the Hamiltonian drift matrix (modifies the passed array)
        for m in range(M):
            for n in range(N):
                xmn = X[m, n]
                s = (1j * 0.5 * U * m * (m - 1.0) - k*n) * xmn
                if n < N-1:
                    xmn1 = X[m, n + 1]
                    s -= epsilon * sqrt_n1[n] * xmn1
                    bx += xmn * (xmn1.real - 1j * xmn1.imag) * sqrt_n1[n]
                if n > 0:
                    s += epsilon * sqrt_n[n] * X[m, n - 1]
                if m > 0 and n < N-1:
                    s += Omega * sqrt_m_n1[m, n] * X[m - 1, n + 1]
                if m < M-1 and n > 0:
                    s -= Omega * sqrt_m1_n[m, n] * X[m + 1, n - 1]
                BX_coherent[m, n] = s

                 # Calculate the drift scalar bx
                norm += xmn.real * xmn.real + xmn.imag * xmn.imag
        bx = np.sqrt(2 * k) * bx / norm
        bx_scalar[0] = 1.0 * bx

        # Calculate the dissipative drift matrix and the total drift (modifies the passed array)
        for m in range(M):
            for n in range(N):
                if n < N - 1:
                    s2 = np.sqrt(2) * bx * sqrt_k_n1[n] * X[m,n+1]
                else:
                    s2 = 0.0 + 0.0j
                
                BX_noncoherent[m,n] = s2
                # Calculates total drift matrix
                BX[m, n] = BX_coherent[m, n] + s2


    @staticmethod
    @njit(fastmath=True)
    def calculate_noise_matrix(X, ZX, 
                    M, N, k, Omega, epsilon, U, 
                    sqrt_n, sqrt_n1, sqrt_m_n1, sqrt_m1_n, sqrt_k_n1):
        '''
        Calculates the noise matrix taking as input a 
        precomputed square root array.

        X: Probability amplitude (Matrix of C coefficients)
        ZX: Resulting noise matrix
        '''
        for m in range(M):
            for n in range(N):
                if n < N - 1:
                    ZX[m,n] = np.sqrt(2) * sqrt_k_n1[n] * X[m,n+1]  #MODIFIED TEST
                else:
                    ZX[m,n] = 0.0 + 0.0j
    

    @staticmethod
    @njit
    def euler_step_current_old(alpha, z, dt, bx_scalar, k, kfill):
        '''Calculates current observable to enable comparison with
        experimental data using Euler method, z should correspond 
        to the complex valued noise used to calculate the noise matrix ZX.
        Note bx_scalar must be evaluated a start of time step'''
        dq = bx_scalar[0] * dt + np.sqrt(dt) * z 
        alpha -= 0.5 * kfill * (dt * alpha - np.sqrt(k) * dq)
        return alpha
    
    @staticmethod
    @njit
    def euler_step_current(alpha, z, dt, bx_scalar, k, kfill):
        '''Calculates current observable to enable comparison with
        experimental data using Euler method, z should correspond 
        to the complex valued noise used to calculate the noise matrix ZX.
        Note bx_scalar must be evaluated a start of time step'''
        dq = bx_scalar[0] * dt + np.sqrt(dt) * z 
        alpha -= 0.5 * kfill * (dt * alpha - dq*np.sqrt(0.5))
        return alpha
    

    @staticmethod
    @njit
    def backward_euler_step_current(alpha, z, dt, bx_scalar, k, kfill):
        '''Calculates current observable to enable comparison with
        experimental data using backward Euler metod, z should correspond 
        to the complex valued noise used to calculate the noise matrix ZX.
        Note bx_scalar must be evaluated a end of time step'''
        dq = bx_scalar[0] * dt + np.sqrt(dt) * z 
        denom = (1 + 0.5 * kfill *dt)
        alpha = (alpha + 0.5 * kfill * np.sqrt(0.5*k) * dq)/denom
        return alpha
    

    