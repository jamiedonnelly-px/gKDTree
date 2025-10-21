from cudaKDTree import cuKNN
import numpy as np
from time import time
from scipy.spatial import KDTree
from tqdm import tqdm

def test():
    N, q, k = 2**18, 2**22, 64
    points, queries = np.random.randn(N, 3), np.random.randn(q, 3)
    start = time()
    gpu_indices, _ = gdk.knn_cuda(points, queries, k)
    end = time()
    print(f"Took {end-start:.3f}s on GPU")

    start = time()
    tree = KDTree(points)
    _, cpu_indices = tree.query(queries, k=k)
    end = time()
    print(f"Took {end-start:.3f}s on CPU")

    for i in tqdm(range(100)):
        assert np.all(sorted(cpu_indices[i]) == sorted(gpu_indices[i]))
    print(f"First {100} indices match perfectly.")


if __name__=="__main__":
    test()
