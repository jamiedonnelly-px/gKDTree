# gKDTree

A GPU-accelerated K-nearest neighbors library for 3D point clouds using CUDA.

## Usage 

Given some observed points, $P\in\mathbb{R}^{N\times 3}$, to find the `K` nearest neighbours for a set of query points, $Q\in\mathbb{R}^{M\times 3}$, and execute the query on the GPU.

NOTE: The `cuda` (GPU) backend can only run queries on sets of points in 3D space. 

```python
from gKDTree import knn_search_3d
distances, indices = knn_search_3d(P, Q, K, device="cuda")
```

Where the value of K must be in a predefined set of allowed values:
```python
K = [1, 2, 4, 8, 16, 32, 64, 128, 256]
```

The function returns:
- `distances`: array of floats with shape `[M, K]` containing the ordered Euclidean distances 
- `indices`: array of integers with shape `[M, K]` containing the corresponding point indices

For each query point `i`, `distances[i]` contains the K nearest distances in ascending order, and `indices[i]` contains the corresponding indices into the original point set `P`.

## Backends
There is an additional CPU-only (default) backend that can be used with
```python
from gKDTree import knn_search_3d
distances, indices = knn_search_3d(P, Q, K, device="cpu")
```
This behaviour will default to running `query` on a `scipy.spatial.KDTree` data structure.  

NOTE: Despite the function naming, when running with the `cpu` backend, queries can be run in arbitrary dimensions, not limited to 3D.

## Installation
The package can be installed by cloning the repository and running:
```bash
pip install .
```

By default this build will check registered precompiled binaries for the system and automatically select one that matches (determined by CUDA version in `nvidia-smi`). If you want to build from source then set the environment variable `FORCE_BUILD_FROM_SOURCE=1` before building e.g, `FORCE_BUILD_FROM_SOURCE=1 pip install git+https://github.com/your-username/gKDTree.git`.

Similarly, if you do not want to the use the CUDA functionality at all and just want a CPU-only package, you can set `FORCE_CPU_ONLY=1` to skip checking for a precompiled binary and building from source. 

### CUDA Configuration
To specify which CUDA toolkit to build against, set one of the following environment variables:
- `NVCC_PATH`: Direct path to the CUDA compiler (e.g., `/usr/local/cuda-12/bin/nvcc`)  
- `CUDA_HOME`: Path to CUDA installation directory (e.g., `/usr/local/cuda-12`)

If neither is set, the build will search for a compiler at `/usr/local/cuda/bin` by default.

## Requirements

- **CMake**: 3.24+ (for optimal GPU architecture detection)
- **CUDA Toolkit**: 11.0 or later
- **Python**: 3.10-3.12
  - NumPy: 1.24 - 1.26 (exclusive)
  - SciPy: 1.10.1 - 1.16 (exclusive)
- **Optional**: OpenMP (automatically detected and used if available)

## GPU Architecture Support

The library automatically builds for all major GPU architectures supported by your CUDA toolkit, including modern GPUs from the RTX 20xx/30xx/40xx series and Tesla/A100/H100 compute cards.