import numpy as np
import gauss
from typing import Callable
import functools

np.random.seed(0)
n = 32
M1 = np.random.rand(n, n)
M2 = np.random.rand(n, n)
M3 = np.random.rand(n, n)
n = 128
H = np.random.rand(n, n)
matrices = [M1, M2, M3, H]

def all_tests(method: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]):
    # Test on real matrices.
    for M in matrices:
        L, U = method(M.copy())
        print(L.shape, U.shape, M.shape)
        assert np.allclose(np.tril(L), L)
        assert np.allclose(np.triu(U), U)
        assert np.allclose(M, L @ U)

# ----- Unit tests ------
def test_basic_gaussian_elimination():
    all_tests(gauss.basic_gaussian_elimination)

def test_blas_2_gaussian_elimination():
    all_tests(gauss.blas_2_gaussian_elimination)

def test_block_size_32_gaussian_elimination():
    all_tests(functools.partial(gauss.block_gaussian_elimination, 4))

# ----- Benchmarks -----
def test_basic_benchmark(benchmark):
    benchmark(gauss.basic_gaussian_elimination, H)

def test_blas_2_benchmark(benchmark):
    benchmark(gauss.blas_2_gaussian_elimination, H)

def test_block_size_32_benchmark(benchmark):
    benchmark(gauss.block_gaussian_elimination, 32, H)