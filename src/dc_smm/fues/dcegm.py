
import os
import numpy as np
from numba import jit
import time
import dill as pickle
from numba import njit, prange

# LinearInterp import removed - was unused
#from HARK.dcegm import calc_segments, calc_multiline_envelope, calc_cross_points
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope, calc_linear_crossing
from interpolation import interp

# Control verbose output with environment variable
DCEGM_VERBOSE = os.environ.get("DCEGM_VERBOSE", "0") == "1"

 # Plot them, and store them as [m, v] pairs

def dcegm(c, dela, vf, a_prime, x, verbose=None):
    
    #(vf)

    start, end = calc_nondecreasing_segments(x, x)
    
    # Debug: print segment info (use env var if verbose not explicitly set)
    if verbose is None:
        verbose = DCEGM_VERBOSE
    if verbose:
        print(f"[DCEGM] Grid size: {len(x)}, Non-decreasing segments: {len(start)}")
    
    segments = []
    c_segments = []
    a_segments = []
    m_segments = []
    v_segments = []
    dela_segments = []

    

    for j in range(len(start)):
        idx = range(start[j], end[j] + 1)
        segments.append([x[idx], vf[idx]])
        c_segments.append(c[idx])
        a_segments.append(a_prime[idx])
        m_segments.append(x[idx])
        v_segments.append(vf[idx])
        dela_segments.append(dela[idx])
    
    #print("a_segments", m_segments)

    m_upper, v_upper, inds_upper = upper_envelope(segments, calc_crossings=False)
    # Use np.full instead of zeros_like + nan (avoids creating 2 arrays per output)
    n_upper = len(m_upper)
    c1_env = np.full(n_upper, np.nan)
    a1_env = np.full(n_upper, np.nan)
    v1_env = np.full(n_upper, np.nan)
    d1_env = np.full(n_upper, np.nan)

    for k, c_segm in enumerate(c_segments):
        c1_env[inds_upper == k] = c_segm[m_segments[k].searchsorted(
            m_upper[inds_upper == k])]

    for k, a_segm in enumerate(a_segments):
        a1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k],
                                            m_segments[k], a_segm)
    #print(v_segments)
    for k, v_segm in enumerate(v_segments):
        v1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k], m_segments[k], v_segm)
    
    for k, dela_segm in enumerate(dela_segments):
        d1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k], m_segments[k], dela_segm)

    # Note: LinearInterp was previously created here but never used - removed
    indices = np.where(np.in1d(a1_env, a_prime))[0]
    a1_env2 = a1_env[indices]
    m_upper2 = m_upper[indices]
    c_env2 = c1_env[indices]
    v_env2 = v1_env[indices]
    d_env2 = d1_env[indices]

    return a1_env2, m_upper2,c_env2, v_env2,d_env2
