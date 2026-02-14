import cupy as cp
from pcg import pcg
import time

n = 5000
A = cp.random.randn(n, n, dtype=cp.float32)
rhs = cp.random.randn(n, 1, dtype=cp.float32)
x0 = cp.zeros((n, 1), dtype=cp.float32)

start = time.perf_counter()
x, its = pcg(A, rhs, x0, m=100, tol=1e-6)
cp.cuda.runtime.deviceSynchronize()
end = time.perf_counter()

print(f"GPU PCG runtime: {end - start:.3f}s | Iterations: {its}")
