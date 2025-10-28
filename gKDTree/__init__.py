import logging
from typing import Literal
import numpy as np
from scipy.spatial import KDTree
import warnings

LOGGER = logging.getLogger(__name__) 

# Try to import CUDA backend
CPU_ONLY = True
try:
    from ._internal import knn_cuda
    CPU_ONLY = False
    LOGGER.info("CUDA backend loaded successfully")
except ImportError as e:
    LOGGER.warning(f"CUDA backend not available: {e}. Falling back to CPU-only mode.")
    # Check if it's a CUDA runtime issue vs missing binary
    try:
        import subprocess
        subprocess.run(['nvidia-smi'], check=True, capture_output=True)
        LOGGER.info("NVIDIA GPU detected but CUDA backend unavailable. "
                   "Consider installing CUDA-enabled version.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        LOGGER.debug("No NVIDIA GPU detected - CPU fallback is appropriate.")

K_VALUES = [2**i for i in range(9)] # 1 -> 256

__all__ = ["knn_search", "get_backend", "is_cuda_available"]

def _cpu_knn(points: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform k-nearest neighbor search using CPU-based scipy KDTree.
    
    Args:
        points: Array of shape (N, 3) containing reference points
        queries: Array of shape (M, 3) containing query points  
        k: Number of nearest neighbors to find
        
    Returns:
        Tuple of (distances, indices) where:
        - distances: Array of shape (M, k) with distances to k nearest neighbors
        - indices: Array of shape (M, k) with indices of k nearest neighbors
    """
    tree = KDTree(points)
    return tree.query(queries, k=k)


def _gpu_knn(points: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform k-nearest neighbor search using CUDA GPU acceleration.
    
    Args:
        points: Array of shape (N, 3) containing reference points
        queries: Array of shape (M, 3) containing query points
        k: Number of nearest neighbors to find
        
    Returns:
        Tuple of (distances, indices) where:
        - distances: Array of shape (M, k) with distances to k nearest neighbors  
        - indices: Array of shape (M, k) with indices of k nearest neighbors
    """
    indices, distances = knn_cuda(points, queries, k)
    return distances, indices


def knn_search_3d(
    points: np.ndarray, 
    queries: np.ndarray, 
    k: int, 
    backend: Literal["cuda", "cpu"] = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find k-nearest neighbors in 3D euclidean space using GPU acceleration.
    
    This function provides a unified interface for k-nearest neighbor search,
    supporting both CPU-based scipy KDTree and CUDA GPU acceleration.
    
    Fallback backend of CPU defaults to standard scipy.spatial.KDTree behaviour
    and can predict in arbitrary dimensions.

    Args:
        points: Reference points array of shape (N, 3). These are the points
            that will be searched for nearest neighbors.
        queries: Query points array of shape (M, 3). For each query point,
            the k nearest neighbors from `points` will be found.
        k: Number of nearest neighbors to find for each query point.
            Must be a positive integer.
        backend: Computation backend to use. Either "cpu" for scipy KDTree
            or "cuda" for CUDA acceleration. Defaults to "cpu".
            
    Returns:
        Tuple of (distances, indices) where:
        - distances: Array of shape (M, k) containing distances from each
            query point to its k nearest neighbors
        - indices: Array of shape (M, k) containing indices into `points`
            array for the k nearest neighbors of each query point
            
    Raises:
        AssertionError: If input validation fails:
            - k is not a pre-approved integer
            - points or queries are not 2D arrays
            - points or queries don't have exactly 3 columns
            - device is not "cpu" or "gpu"
            
    Example:
        >>> points = np.random.rand(1000, 3)
        >>> queries = np.random.rand(100, 3) 
        >>> distances, indices = cuNN(points, queries, k=5, device="gpu")
        >>> print(f"Found {len(indices)} query results with {indices.shape[1]} neighbors each")
    """
    # Validate backend
    if CPU_ONLY and backend == "cuda":
        raise RuntimeError(f"Current backend ({get_backend()}) does not support GPU acceleration.")

    # Input validation
    if not isinstance(k, int):
        raise AssertionError(f"k must be of type int, received type: {type(k)}")
    
    if k not in K_VALUES:
        raise AssertionError(f"k must be one of {K_VALUES}, received: {k}")
    
    if len(points.shape) != 2 or len(queries.shape) != 2:
        raise AssertionError(
            f"points and queries must both be 2-dimensional arrays. "
            f"Received shapes {points.shape} and {queries.shape}, respectively."
        )
    
    if (points.shape[1] != 3 or queries.shape[1] != 3) and not CPU_ONLY:
        raise AssertionError(
            f"Fpoints and queries must be shape (N, 3) and (M, 3) for CUDA backend. "
            f"Received shapes {points.shape} and {queries.shape}, respectively."
        )
    
    # Dispatch to appropriate backend
    match backend:
        case "cpu":
            LOGGER.debug(f"Using CPU KDTree for {len(queries)} queries against {len(points)} points")
            return _cpu_knn(points, queries, k)
        case "gpu":
            LOGGER.debug(f"Using GPU CUDA for {len(queries)} queries against {len(points)} points")
            return _gpu_knn(points, queries, k)
        case _:
            raise AssertionError(f"device must be either 'cpu' or 'gpu', received: '{backend}'")

def is_cuda_available() -> bool:
    """Check if CUDA backend is available"""
    return not CPU_ONLY

def get_backend() -> str:
    """Get current backend"""
    return "cpu" if CPU_ONLY else "cuda"

if CPU_ONLY:
    warnings.warn(
        "gKDTree is running in CPU-only mode. For GPU acceleration, "
        "ensure CUDA is installed and reinstall the package.",
        UserWarning,
        stacklevel=2
    )