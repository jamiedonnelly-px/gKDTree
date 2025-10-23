# gKDTree

A GPU-accelerated K-nearest neighbors library for 3D point clouds using CUDA.

## Usage 

Given some observed points, $P\in\mathbb{R}^{N\times 3}$, to find the `K` nearest neighbours for a set of query points, $Q\in\mathbb{R}^{M\times 3}$, and execute the query on the GPU:

```python
from gKDTree import cuNN
distances, indices = cuNN(P, Q, K, device="gpu")
```

Where the value of K must be in a predefined set of allowed values:
```python
K = [1, 2, 4, 8, 16, 32, 64, 128, 256]
```

The function returns:
- `distances`: array of floats with shape `[M, K]` containing the ordered Euclidean distances 
- `indices`: array of integers with shape `[M, K]` containing the corresponding point indices

For each query point `i`, `distances[i]` contains the K nearest distances in ascending order, and `indices[i]` contains the corresponding indices into the original point set `P`.

## Installation

The package can be installed by cloning the repository and running:
```bash
pip install .
```

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