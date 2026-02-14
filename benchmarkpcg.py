import time
import cupy as cp
from pcg import pcg
#import numpy as np
#from pcg_cpu import pcg  # numpy version
#on cpu version
n = 5000          # start small
m = 1000
tol = 1e-8

# SPD matrix
A = cp.random.randn(n, n, dtype=cp.float32)
A = A.T @ A + 1e-3 * cp.eye(n)

rhs = cp.random.randn(n, dtype=cp.float32)
x0 = cp.zeros_like(rhs)

# warm-up (important)
pcg(A, rhs, x0, 5, tol)
cp.cuda.runtime.deviceSynchronize()

# timed run
start = time.perf_counter()
x, its = pcg(A, rhs, x0, m, tol)
cp.cuda.runtime.deviceSynchronize()
end = time.perf_counter()

print(f"GPU PCG time: {end - start:.3f}s")
print(f"Iterations: {its}")
