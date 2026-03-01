""" 
Simple implementation of the roof-top cut algorithm without iterative search
and without intersection points. 

See Dobrescru and Shanker, 2024, "Discrete-Continuous High Dimensional Dynamic Programming"

"""

import numpy as np
from pykdtree.kdtree import KDTree as KDTree



def _rfc_vectorized(M, gradM, Qval, sigma, J_bar, radius, closest_indices, dd, k):
    """Helper function to compute sub-optimal points in a vectorized manner."""
    n, d = M.shape  # Number of points and dimension of grid
    dp = sigma.shape[1]  # Number of policies

    # Evaluate values on tangent plane and check which points are below
    gradrs = gradM.reshape(n, d, 1)
    Md = M[closest_indices].reshape(n * (k - 1), d) - np.repeat(M, k - 1, axis=0)
    Mdn = Md.reshape(n, k - 1, d)
    tngv = (Mdn @ gradrs).reshape(n, k - 1)
    vdiff = Qval[closest_indices].reshape(n, k - 1) - Qval
    I1 = vdiff < tngv  # Indicator for points below tangent plane

    # Check for jump in policy
    Pd = sigma[closest_indices].reshape(n * (k - 1), dp) - np.repeat(sigma, k - 1, axis=0)
    Pdnorm = np.linalg.norm(Pd, axis=1).reshape(n, k - 1)
    delsig = np.abs(Pdnorm / np.linalg.norm(Md, axis=1).reshape(n, k - 1))
    I2 = delsig > J_bar  # Indicator for significant policy change

    # Only include neighbours within radius
    I3 = dd[:, 1:] < radius

    # Select sub-optimal points
    sub_optimal_points = np.unique(np.where(I1 * I2 * I3, closest_indices + 1, 0))
    sub_optimal_points = sub_optimal_points[1:] - 1

    return sub_optimal_points, tngv, closest_indices


def rfc(M, gradM, Qval, sigma, J_bar, radius, k):
    """Implements vectorized version of roof-top cut and eliminates points below 
    the tangent plane and with jumps policy (grid distance function).

    Parameters:
    M : ndarray
        Irregular grid.
    gradM : ndarray
        Gradient of value function at each point on the grid (each column represents a dimension).
    Qval : ndarray
        Value function at each point on the grid.
    sigma : ndarray
        Policy function at each point on the grid (each policy represented as a column).
    J_bar : float
        Jump detection threshold.
    radius : float
        Neighbour distance threshold.
    k : int
        Number of neighbours to consider.

    Returns:
    sub_optimal_points : ndarray
        Indices of sub-optimal points.
    tngv : ndarray
        Tangent plane values for neighbouring points.
    closest_indices : ndarray
        Indices for neighbouring points.
    """

    tree = KDTree(M)
    dd, closest_indices = tree.query(M, k)
    closest_indices = closest_indices[:, 1:]  # Keep only strict neighbours
    closest_indices[closest_indices>=M.shape[0]] = M.shape[0] - 1  # Avoid out of bounds

    return _rfc_vectorized(M, gradM, Qval, sigma, J_bar, radius, closest_indices, dd, k)