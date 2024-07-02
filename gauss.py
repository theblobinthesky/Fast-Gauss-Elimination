import numpy as np
from scipy.linalg.lapack import dtrtri

# This will be needed in all routines...
def unpack(A: np.ndarray)->tuple[np.ndarray, np.ndarray]:
    m, n = A.shape
    assert m >= n

    # Extract L and U.
    L = np.tril(A).copy()
    L[range(n), range(n)] = 1.0
    U = np.triu(A)[:n, :n].copy()
    
    return L, U

def basic_gaussian_elimination(A: np.ndarray)->tuple[np.ndarray, np.ndarray]:
    m, n = A.shape[0], A.shape[1]

    for k in range(min(m - 1, n)):
        # L bearbeiten.
        for j in range(k+1, m):
            A[j, k] /= A[k, k]

        # U bearbeiten. Beim letzten Schritt nicht arbeiten.
        if k < n - 1:
            for j in range(k+1, m):
                for i in range(k+1, n):
                    A[j, i] -= A[j, k] * A[k, i]

    return unpack(A)

def blas_2_inplace_gaussian_elimination(A: np.ndarray):
    m, n = A.shape[0], A.shape[1]

    for k in range(min(m - 1, n)):
        # L bearbeiten.
        A[k+1:, k] /= A[k, k]

        # U bearbeiten.
        if k < n:
            A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])

def blas_2_gaussian_elimination(A: np.ndarray)->tuple[np.ndarray, np.ndarray]:
    blas_2_inplace_gaussian_elimination(A)
    return unpack(A)

def old_block_gaussian_elimination(b: int, A: np.ndarray)->tuple[np.ndarray, np.ndarray]:
    m = A.shape[0]
    assert m == A.shape[1]
    assert m % b == 0

    for k in range(0, m - 1, b):
        # Faktorisiere Teilmatrix A hut.
        L, _ = blas_2_gaussian_elimination(A[k:, k:k+b])
        L_1, L_2 = L[:b, :], L[b:, :]

        # Erstelle U_23 in A.
        A[k:k+b,k+b:] = np.linalg.inv(L_1) @ A[k:k+b,k+b:]

        # Erstelle A hut in A.
        A[k+b:, k+b:] = A[k+b:, k+b:] - L_2 @ A[k:k+b,k+b:]

    return unpack(A)

def block_gaussian_elimination(b: int, A: np.ndarray)->tuple[np.ndarray, np.ndarray]:
    m = A.shape[0]
    assert m == A.shape[1]
    assert m % b == 0

    for k in range(0, m - 1, b):
        # Faktorisiere Teilmatrix A hut.
        X = A[k:, k:k+b]
        # Kopiere nur L und nicht U und spare somit etwas Zeit.
        blas_2_inplace_gaussian_elimination(X)
        L = np.tril(X).copy()
        np.fill_diagonal(L, 1.0)
        L_1, L_2 = L[:b, :], L[b:, :]

        # Erstelle U_23 in A.
        L_1_inv, _ = dtrtri(L_1, lower=1)
        A[k:k+b,k+b:] = L_1_inv @ A[k:k+b,k+b:]

        # Erstelle A hut in A.
        A[k+b:, k+b:] = A[k+b:, k+b:] - L_2 @ A[k:k+b,k+b:]

    return unpack(A)