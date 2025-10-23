from gKDTree import cuNN
import numpy as np
from time import time
from tqdm import tqdm

def test():
    N, q, k = 2**18, 2**22, 64
    points, queries = np.random.randn(N, 3), np.random.randn(q, 3)
    start = time()
    _, gpu_indices = cuNN(points, queries, k, device="gpu")
    end = time()
    print(f"Took {end-start:.3f}s on GPU", flush=True)

    start = time()
    _, cpu_indices = cuNN(points, queries, k, device="cpu")
    end = time()
    print(f"Took {end-start:.3f}s on CPU", flush=True)

    for i in tqdm(range(100)):
        assert np.all(sorted(cpu_indices[i]) == sorted(gpu_indices[i]))
    print(f"First {100} indices match perfectly.", flush=True)


if __name__=="__main__":
    test()