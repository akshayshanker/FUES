# FUES Algorithm Updates - August 2025

## Overview
Recent development focused on resolving numerical stability issues and improving the robustness of intersection calculations in the Fast Upper-Envelope Scan (FUES) algorithm. The changes address critical edge cases where the algorithm previously failed to maintain envelope continuity, particularly around policy function jumps and near-parallel segments.

## Key Technical Improvements

### 1. Robust Intersection Handling
**Problem:** The algorithm struggled with near-parallel line segments where traditional intersection calculations became numerically unstable, leading to gaps in the computed envelope.

**Solution:** Implemented a forced intersection framework (`_force_crossing_inside`) that guarantees valid intersection points even for challenging geometries:
- Uses parametric line representation instead of slope-based calculations
- Implements adaptive epsilon separation based on interval size
- Clips intersection coordinates to ensure they remain strictly within valid bounds
- Averages y-values from both branches to minimize numerical drift

### 2. Branch Continuity Detection
**Problem:** The algorithm needed to distinguish between points on the same policy branch versus those separated by discrete jumps, critical for correct envelope construction.

**Solution:** Added intelligent branch detection (`check_same_branch`) using two criteria:
- Gradient threshold test comparing policy derivatives against configurable `m_bar` parameter
- Distance constraint ensuring points are sufficiently close (within `EPS_fwd_back`)
- Safe extrapolation point finder for cases where direct neighbors aren't suitable

### 3. Enhanced Forward and Backward Scanning
**Problem:** Previous implementation could fail to find appropriate bracketing points for intersection calculations, especially after policy jumps.

**Solution:** Refined the scanning logic with multiple improvements:
- Backward scan now searches through circular buffer of recently dropped points
- Forward scan validates jumps by checking both value conditions and gradient thresholds
- Added fallback extrapolation when standard bracketing fails
- Uniform index bookkeeping across all cases for better maintainability

### 4. Consecutive Jump Prevention
**Problem:** Multiple consecutive policy jumps could create numerical instabilities and violate envelope properties.

**Solution:** Implemented "no two jumps" rule with intelligent enforcement:
- Tracks jump history through iterations
- When consecutive jump detected, drops previously jumped-to point
- Removes associated intersections to maintain consistency
- Only enforces rule when current jump is validated and kept

## Performance Optimizations

### Numerical Stability Constants
Adjusted epsilon values for improved numerical behavior:
- `EPS_D = 1e-20`: Division protection (reduced from 1e-200)
- `EPS_A = 1e-20`: Gradient calculations
- `EPS_SEP = 1e-10`: Intersection separation
- `EPS_fwd_back = 10`: Forward/backward scan distance threshold
- `PARALLEL_GUARD = 1e-12`: Near-parallel line detection

### Memory Efficiency
- Pre-allocated intersection arrays sized at `2*(N-1)` to handle worst-case scenarios
- Circular buffer implementation for backward scanning reduces memory churn
- Eliminated redundant array allocations in hot loops

## Algorithm Variants

### Current Production Version (`fues.py`)
The main implementation incorporating all stability improvements, suitable for production use with housing-renting models.

### Original Paper Version (`fues_v0dev.py`)
Preserved the October 2024 paper implementation for reproducibility and comparison studies.

### Experimental Versions
Multiple development variants exploring different approaches to intersection handling and scan logic, maintained in `experimental/` directory for ongoing research.

## Impact on Housing Model Results

Testing with the housing-renting model shows:
- Elimination of spurious Euler equation residuals at policy function kinks
- Improved convergence stability for high-resolution grids
- Consistent envelope construction across different parameter configurations
- Better performance with complex multi-dimensional state spaces

## Configuration Recommendations

For optimal results with current implementation:
- Use `m_bar` around 1.0-1.5 for typical housing models
- Enable intersection computation for problems with discrete choices
- Monitor debug output in regions of interest when troubleshooting
- Consider adaptive grid refinement near policy function jumps

## Future Development

Areas for continued improvement:
- Adaptive `m_bar` selection based on local problem characteristics
- GPU acceleration of intersection calculations
- Automatic detection of problematic regions for targeted refinement
- Integration with machine learning for parameter tuning