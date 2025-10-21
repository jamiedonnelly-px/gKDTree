"""CUDA-accelerated k-nearest neighbors implementation."""
import logging
from typing import Literal, get_args
import numpy as np

from ._cuKDTree import knn_cuda

LOGGER = logging.getLogger(__file__)

KLiterals = Literal[1, 2, 3, 4, 5, 8, 10, 16, 32, 50, 64]

def cuKNN(points: np.ndarray, queries: np.ndarray, k: KLiterals) -> tuple[np.ndarray[int], np.ndarray[float]]:
    """
    Find k-nearest neighbors using CUDA acceleration.
    
    Args:
        points: Reference points array of shape (N, 3)
        queries: Query points array of shape (M, 3) 
        k: Number of nearest neighbors to find
        
    Returns:
        Tuple of (indices, distances) where:
        - indices: Array of shape (M, k) with neighbor indices
        - distances: Array of shape (M, k) with neighbor distances
        
    Raises:
        ValueError: If input format or shape is not supported
        RuntimeError: If CUDA operation fails
        
    Example:
        >>> points = np.random.rand(1000, 3).astype(np.float32)
        >>> queries = np.random.rand(100, 3).astype(np.float32) 
        >>> indices, distances = cuKNN(points, queries, k=5)
    """
    if len(points.shape) != 2 or points.shape[-1] != 3:
        raise ValueError(f"Points must be of shape (N, 3), received: {points.shape}")
    if len(queries.shape) != 2 or queries.shape[-1] != 3:
        raise ValueError(f"Queries must be of shape (M, 3), received: {queries.shape}")
    valid_k_values = get_args(KLiterals)
    if k not in valid_k_values:
        raise ValueError(f"k must be one of {valid_k_values}, received: {k}")
    
    try:
        indices, distances = knn_cuda(points, queries, k)
    except RuntimeError as e:
        raise e

    return indices, distances