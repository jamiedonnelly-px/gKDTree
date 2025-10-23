import logging
from typing import Literal
import numpy as np
from scipy.spatial import KDTree
from ._internal import knn_cuda

LOGGER = logging.getLogger(__name__)  # Fixed: use __name__ instead of __file__

K_VALUES = [2**i for i in range(9)] # 1 -> 256

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


def cuNN(
    points: np.ndarray, 
    queries: np.ndarray, 
    k: int, 
    device: Literal["cpu", "gpu"] = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find k-nearest neighbors using CPU or GPU acceleration.
    
    This function provides a unified interface for k-nearest neighbor search,
    supporting both CPU-based scipy KDTree and CUDA GPU acceleration.
    
    Args:
        points: Reference points array of shape (N, 3). These are the points
            that will be searched for nearest neighbors.
        queries: Query points array of shape (M, 3). For each query point,
            the k nearest neighbors from `points` will be found.
        k: Number of nearest neighbors to find for each query point.
            Must be a positive integer.
        device: Computation device to use. Either "cpu" for scipy KDTree
            or "gpu" for CUDA acceleration. Defaults to "cpu".
            
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
    
    if points.shape[1] != 3 or queries.shape[1] != 3:  # Fixed: was checking points.shape[-1] twice
        raise AssertionError(
            f"points and queries must be shape (N, 3) and (M, 3). "
            f"Received shapes {points.shape} and {queries.shape}, respectively."
        )
    
    # Dispatch to appropriate backend
    match device:
        case "cpu":
            LOGGER.debug(f"Using CPU KDTree for {len(queries)} queries against {len(points)} points")
            return _cpu_knn(points, queries, k)
        case "gpu":
            LOGGER.debug(f"Using GPU CUDA for {len(queries)} queries against {len(points)} points")
            return _gpu_knn(points, queries, k)  # Fixed: was calling _cpu_knn
        case _:
            raise AssertionError(f"device must be either 'cpu' or 'gpu', received: '{device}'")