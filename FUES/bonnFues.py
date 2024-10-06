"""Numba implementation of the Fast Upper-Envelope Scan (FUES) by Open Source Economics:

https://github.com/OpenSourceEconomics/upper-envelope

The original FUES algorithm is based on Loretti I. Dobrescu and Akshay Shanker (2022)
'Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming',
https://dx.doi.org/10.2139/ssrn.4181302



"""

from typing import Callable, Optional, Tuple

import numpy as np
from numba import njit


@njit
def fues_numba(
    endog_grid: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    expected_value_zero_savings: np.ndarray | float,
    value_function: Callable,
    value_function_args: Tuple,
    n_constrained_points_to_add=None,
    n_final_wealth_grid=None,
    jump_thresh=2,
    n_points_to_scan=10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop suboptimal points and refine the endogenous grid, policy, and value.

    Computes the upper envelope over the overlapping segments of the
    decision-specific value functions, which in fact are value "correspondences"
    in this case, where multiple solutions are detected. The dominated grid
    points are then eliminated from the endogenous wealth grid.
    Discrete choices introduce kinks and non-concave regions in the value
    function that lead to discontinuities in the policy function of the
    continuous (consumption) choice. In particular, the value function has a
    non-concave region where the decision-specific values of the
    alternative discrete choices (e.g. continued work or retirement) cross.
    These are referred to as "primary" kinks.
    As a result, multiple local optima for consumption emerge and the Euler
    equation has multiple solutions.
    Moreover, these "primary" kinks propagate back in time and manifest
    themselves in an accumulation of "secondary" kinks in the choice-specific
    value functions in earlier time periods, which, in turn, also produce an
    increasing number of discontinuities in the consumption functions
    in earlier periods of the life cycle.
    These discontinuities in consumption rules in period t are caused by the
    worker's anticipation of landing exactly at the kink points in the
    subsequent periods t + 1, t + 2, ..., T under the optimal consumption policy.

    Args:
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific endogenous grid.
        policy (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific policy function.
        value (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific value function.
        expected_value_zero_savings (np.ndarray | float): The agent's expected value
            given that she saves zero.
        value_function (callable): The value function for calculating the value if
            nothing is saved.
        value_function_args (Tuple): The positional arguments to be passed to the value
            function.
        n_constrained_points_to_add (int): Number of constrained points to add to the
                left of the first grid point if there is an area with credit-constrain.
        n_final_wealth_grid (int): Size of final function grid.
        jump_thresh (float): Jump detection threshold.
        n_points_to_scan (int): Number of points to scan for suboptimal points.
    Returns:
        tuple:

        - endog_grid_refined (np.ndarray): 1d array of shape (1.1 * n_grid_wealth,)
            containing the refined state- and choice-specific endogenous grid.
        - policy_refined_with_nans (np.ndarray): 1d array of shape (1.1 * n_grid_wealth)
            containing refined state- and choice-specificconsumption policy.
        - value_refined_with_nans (np.ndarray): 1d array of shape (1.1 * n_grid_wealth)
            containing refined state- and choice-specific value function.

    """
    min_wealth_grid = np.min(endog_grid)

    if endog_grid[0] > min_wealth_grid:
        # Non-concave region coincides with credit constraint.
        # This happens when there is a non-monotonicity in the endogenous wealth grid
        # that goes below the first point.
        # Solution: Value function to the left of the first point is analytical,
        # so we just need to add some points to the left of the first grid point.

        # Set default of n_constrained_points_to_add to 10% of the grid size
        n_constrained_points_to_add = (
            endog_grid.shape[0] // 10
            if n_constrained_points_to_add is None
            else n_constrained_points_to_add
        )

        endog_grid, value, policy = _augment_grids(
            endog_grid=endog_grid,
            value=value,
            policy=policy,
            min_wealth_grid=min_wealth_grid,
            n_constrained_points_to_add=n_constrained_points_to_add,
            value_function=value_function,
            value_function_args=value_function_args,
        )

    endog_grid = np.append(0, endog_grid)
    policy = np.append(0, policy)
    value = np.append(expected_value_zero_savings, value)

    endog_grid_refined, value_refined, policy_refined = fues_numba_unconstrained(
        endog_grid,
        value,
        policy,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
    )

    # Set default value of final grid size to 1.2 times current if not defined
    n_final_wealth_grid = (
        int(1.2 * (len(policy))) if n_final_wealth_grid is None else n_final_wealth_grid
    )

    # Fill array with nans to fit 10% extra grid points
    endog_grid_refined_with_nans = np.empty(n_final_wealth_grid)
    policy_refined_with_nans = np.empty(n_final_wealth_grid)
    value_refined_with_nans = np.empty(n_final_wealth_grid)
    endog_grid_refined_with_nans[:] = np.nan
    policy_refined_with_nans[:] = np.nan
    value_refined_with_nans[:] = np.nan

    endog_grid_refined_with_nans[: len(endog_grid_refined)] = endog_grid_refined
    policy_refined_with_nans[: len(policy_refined)] = policy_refined
    value_refined_with_nans[: len(value_refined)] = value_refined

    return (
        endog_grid_refined_with_nans,
        policy_refined_with_nans,
        value_refined_with_nans,
    )


@njit
def fues_numba_unconstrained(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    policy2: np.ndarray,
    jump_thresh=2,
    n_points_to_scan=10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove suboptimal points from the endogenous grid, policy, and value function.

    Args:
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth + 1,).
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        policy (np.ndarray): 1d array containing the unrefined policy correspondence
            of shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid
            of shape (n_grid_wealth + 1,).
        jump_thresh (float): Jump detection threshold.
    Returns:
        tuple:

        - endog_grid_refined (np.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing the refined endogenous wealth grid.
        - policy_refined (np.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing refined consumption policy.
        - value_refined (np.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing refined value function.

    """

    endog_grid = endog_grid[np.where(~np.isnan(value))[0]]
    policy = policy[np.where(~np.isnan(value))]
    value = value[np.where(~np.isnan(value))]

    idx_sort = np.argsort(endog_grid, kind="mergesort")
    value = np.take(value, idx_sort)
    policy = np.take(policy, idx_sort)
    endog_grid = np.take(endog_grid, idx_sort)
    exog_grid = endog_grid - policy

    (
        value_clean_with_nans,
        policy_clean_with_nans,
        endog_grid_clean_with_nans,
    ) = scan_value_function(
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        exog_grid=exog_grid,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
    )

    endog_grid_refined = endog_grid_clean_with_nans[
        ~np.isnan(endog_grid_clean_with_nans)
    ]
    value_refined = value_clean_with_nans[~np.isnan(value_clean_with_nans)]
    policy_refined = policy_clean_with_nans[~np.isnan(policy_clean_with_nans)]
    policy2_refined = policy2[~np.isnan(policy_clean_with_nans)]

    return endog_grid_refined, value_refined, policy_refined, policy2_refined


@njit
def scan_value_function(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    exog_grid: np.ndarray,
    jump_thresh: Optional[float] = 2,
    n_points_to_scan: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan the value function to remove suboptimal points and add intersection points.

    Args:
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        policy (np.ndarray): 1d array containing the unrefined policy correspondence
            of shape (n_grid_wealth + 1,).
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid
            of shape (n_grid_wealth + 1,).
        jump_thresh (float): Jump detection threshold.
        n_points_to_scan (int): Number of points to scan for suboptimal points.

    Returns:
        tuple:

        - (np.ndarray): 1d array of shape (n_grid_clean,) containing the refined
            value function. Overlapping segments have been removed and only
            the optimal points are kept.

    """

    value_refined, policy_refined, endog_grid_refined = _initialize_refined_arrays(
        value, policy, endog_grid
    )

    suboptimal_points = np.zeros(n_points_to_scan, dtype=np.int64)

    j = 1
    k = 0

    idx_refined = 2

    for i in range(1, len(endog_grid) - 2):
        if value[i + 1] - value[j] < 0:
            suboptimal_points = _append_index(suboptimal_points, i + 1)

        else:
            # value function gradient between previous two optimal points
            grad_before = (value[j] - value[k]) / (endog_grid[j] - endog_grid[k])

            # gradient with leading index to be checked
            grad_next = (value[i + 1] - value[j]) / (endog_grid[i + 1] - endog_grid[j])

            switch_value_func = (
                np.abs(
                    (exog_grid[i + 1] - exog_grid[j])
                    / (endog_grid[i + 1] - endog_grid[j])
                )
                > jump_thresh
            )

            if grad_before > grad_next and exog_grid[i + 1] - exog_grid[j] < 0:
                suboptimal_points = _append_index(suboptimal_points, i + 1)

            # if right turn is made and jump registered
            # remove point or perform forward scan
            elif grad_before > grad_next and switch_value_func:
                keep_next = False

                (
                    grad_next_forward,
                    idx_next_on_lower_curve,
                    found_next_point_on_same_value,
                ) = _forward_scan(
                    value=value,
                    endog_grid=endog_grid,
                    exog_grid=exog_grid,
                    jump_thresh=jump_thresh,
                    idx_current=j,
                    idx_next=i + 1,
                    n_points_to_scan=n_points_to_scan,
                )

                # get index of closest next point with same discrete choice as point j
                if found_next_point_on_same_value:
                    if grad_next > grad_next_forward:
                        keep_next = True

                if not keep_next:
                    suboptimal_points = _append_index(suboptimal_points, i + 1)
                else:
                    (
                        grad_next_backward,
                        sub_idx_point_before_on_same_value,
                    ) = _backward_scan(
                        value=value,
                        endog_grid=endog_grid,
                        exog_grid=exog_grid,
                        suboptimal_points=suboptimal_points,
                        jump_thresh=jump_thresh,
                        idx_current=j,
                        idx_next=i + 1,
                    )
                    idx_before_on_upper_curve = suboptimal_points[
                        sub_idx_point_before_on_same_value
                    ]

                    intersect_grid, intersect_value = _linear_intersection(
                        x1=endog_grid[idx_next_on_lower_curve],
                        y1=value[idx_next_on_lower_curve],
                        x2=endog_grid[j],
                        y2=value[j],
                        x3=endog_grid[i + 1],
                        y3=value[i + 1],
                        x4=endog_grid[idx_before_on_upper_curve],
                        y4=value[idx_before_on_upper_curve],
                    )

                    intersect_policy_left = _evaluate_point_on_line(
                        x1=endog_grid[idx_next_on_lower_curve],
                        y1=policy[idx_next_on_lower_curve],
                        x2=endog_grid[j],
                        y2=policy[j],
                        point_to_evaluate=intersect_grid,
                    )
                    intersect_policy_right = _evaluate_point_on_line(
                        x1=endog_grid[i + 1],
                        y1=policy[i + 1],
                        x2=endog_grid[idx_before_on_upper_curve],
                        y2=policy[idx_before_on_upper_curve],
                        point_to_evaluate=intersect_grid,
                    )

                    value_refined[idx_refined] = intersect_value
                    policy_refined[idx_refined] = intersect_policy_left
                    endog_grid_refined[idx_refined] = intersect_grid
                    idx_refined += 1

                    value_refined[idx_refined] = intersect_value
                    policy_refined[idx_refined] = intersect_policy_right
                    endog_grid_refined[idx_refined] = intersect_grid
                    idx_refined += 1

                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1

                    k = j
                    j = i + 1

            # if left turn is made or right turn with no jump, then
            # keep point provisionally and conduct backward scan
            else:
                grad_next_backward, sub_idx_point_before_on_same_value = _backward_scan(
                    value=value,
                    endog_grid=endog_grid,
                    exog_grid=exog_grid,
                    suboptimal_points=suboptimal_points,
                    jump_thresh=jump_thresh,
                    idx_current=j,
                    idx_next=i + 1,
                )
                keep_current = True
                current_is_optimal = True
                idx_before_on_upper_curve = suboptimal_points[
                    sub_idx_point_before_on_same_value
                ]

                # # This should better a bool from the backwards scan
                grad_next_forward, _, _ = _forward_scan(
                    value=value,
                    endog_grid=endog_grid,
                    exog_grid=exog_grid,
                    jump_thresh=jump_thresh,
                    idx_current=j,
                    idx_next=i + 1,
                    n_points_to_scan=n_points_to_scan,
                )
                if grad_next_forward > grad_next and switch_value_func:
                    suboptimal_points = _append_index(suboptimal_points, i + 1)
                    current_is_optimal = False

                # if the gradient joining the leading point i+1 (we have just
                # jumped to) and the point m(the last point on the same
                # choice specific policy) is shallower than the
                # gradient joining the i+1 and j, then delete j'th point
                if (
                    grad_before < grad_next
                    and grad_next >= grad_next_backward
                    and switch_value_func
                ):
                    keep_current = False

                if not keep_current and current_is_optimal:
                    intersect_grid, intersect_value = _linear_intersection(
                        x1=endog_grid[j],
                        y1=value[j],
                        x2=endog_grid[k],
                        y2=value[k],
                        x3=endog_grid[i + 1],
                        y3=value[i + 1],
                        x4=endog_grid[idx_before_on_upper_curve],
                        y4=value[idx_before_on_upper_curve],
                    )

                    # The next two interpolations is just to show that from
                    # interpolation from each side leads to the same result
                    intersect_policy_left = _evaluate_point_on_line(
                        x1=endog_grid[k],
                        y1=policy[k],
                        x2=endog_grid[j],
                        y2=policy[j],
                        point_to_evaluate=intersect_grid,
                    )
                    intersect_policy_right = _evaluate_point_on_line(
                        x1=endog_grid[i + 1],
                        y1=policy[i + 1],
                        x2=endog_grid[idx_before_on_upper_curve],
                        y2=policy[idx_before_on_upper_curve],
                        point_to_evaluate=intersect_grid,
                    )

                    if idx_before_on_upper_curve > 0 and i > 1:
                        value_refined[idx_refined - 1] = intersect_value
                        policy_refined[idx_refined - 1] = intersect_policy_left
                        endog_grid_refined[idx_refined - 1] = intersect_grid

                        value_refined[idx_refined] = intersect_value
                        policy_refined[idx_refined] = intersect_policy_right
                        endog_grid_refined[idx_refined] = intersect_grid
                        idx_refined += 1

                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1

                    value[j] = intersect_value
                    policy[j] = intersect_policy_right
                    endog_grid[j] = intersect_grid

                    j = i + 1

                elif keep_current and current_is_optimal and idx_refined< len(endog_grid) - 2:
                    if grad_next > grad_before and switch_value_func:
                        (
                            grad_next_forward,
                            idx_next_on_lower_curve,
                            _,
                        ) = _forward_scan(
                            value=value,
                            endog_grid=endog_grid,
                            exog_grid=exog_grid,
                            jump_thresh=jump_thresh,
                            idx_current=j,
                            idx_next=i + 1,
                            n_points_to_scan=n_points_to_scan,
                        )

                        intersect_grid, intersect_value = _linear_intersection(
                            x1=endog_grid[idx_next_on_lower_curve],
                            y1=value[idx_next_on_lower_curve],
                            x2=endog_grid[j],
                            y2=value[j],
                            x3=endog_grid[i + 1],
                            y3=value[i + 1],
                            x4=endog_grid[idx_before_on_upper_curve],
                            y4=value[idx_before_on_upper_curve],
                        )

                        intersect_policy_left = _evaluate_point_on_line(
                            x1=endog_grid[idx_next_on_lower_curve],
                            y1=policy[idx_next_on_lower_curve],
                            x2=endog_grid[j],
                            y2=policy[j],
                            point_to_evaluate=intersect_grid,
                        )
                        intersect_policy_right = _evaluate_point_on_line(
                            x1=endog_grid[i + 1],
                            y1=policy[i + 1],
                            x2=endog_grid[idx_before_on_upper_curve],
                            y2=policy[idx_before_on_upper_curve],
                            point_to_evaluate=intersect_grid,
                        )

                        value_refined[idx_refined] = intersect_value
                        policy_refined[idx_refined] = intersect_policy_left
                        endog_grid_refined[idx_refined] = intersect_grid
                        idx_refined += 1

                        value_refined[idx_refined] = intersect_value
                        policy_refined[idx_refined] = intersect_policy_right
                        endog_grid_refined[idx_refined] = intersect_grid
                        idx_refined += 1

                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1

                    k = j
                    j = i + 1

    value_refined[idx_refined] = value[-1]
    endog_grid_refined[idx_refined] = endog_grid[-1]
    policy_refined[idx_refined] = policy[-1]

    return value_refined, policy_refined, endog_grid_refined


@njit
def _forward_scan(
    value: np.ndarray,
    endog_grid: np.ndarray,
    exog_grid: np.ndarray,
    jump_thresh: float,
    idx_current: int,
    idx_next: int,
    n_points_to_scan: int,
) -> Tuple[float, int, int]:
    """Scan forward to check whether next point is optimal.

    Args:
        value (np.ndarray): 1d array containing the value function of shape
            (n_grid_wealth + 1,).
        endog_grid (np.ndarray): 1d array containing the endogenous wealth grid of
            shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid of
            shape (n_grid_wealth + 1,).
        jump_thresh (float): Threshold for the jump in the value function.
        idx_current (int): Index of the current point in the value function.
        idx_next (int): Index of the next point in the value function.
        n_points_to_scan (int): The number of points to scan forward.

    Returns:
        tuple:

        - grad_next_forward (float): The gradient of the next point on the same
            value function.
        - is_point_on_same_value (int): Indicator for whether the next point is on
            the same value function.
        - dist_next_point_on_same_value (int): The distance to the next point on
            the same value function.

    """

    is_next_on_same_value = 0
    idx_on_same_value = 0
    grad_next_on_same_value = 0

    idx_max = len(exog_grid) - 1

    for i in range(1, n_points_to_scan + 1):
        idx_to_check = min(idx_next + i, idx_max)
        if endog_grid[idx_current] < endog_grid[idx_to_check]:
            is_on_same_value = (
                np.abs(
                    (exog_grid[idx_current] - exog_grid[idx_to_check])
                    / (endog_grid[idx_current] - endog_grid[idx_to_check])
                )
                < jump_thresh
            )
            is_next = is_on_same_value * (1 - is_next_on_same_value)
            idx_on_same_value = (
                idx_to_check * is_next + (1 - is_next) * idx_on_same_value
            )

            grad_next_on_same_value = (
                (value[idx_next] - value[idx_to_check])
                / (endog_grid[idx_next] - endog_grid[idx_to_check])
            ) * is_next + (1 - is_next) * grad_next_on_same_value

            is_next_on_same_value = (
                is_next_on_same_value * is_on_same_value
                + (1 - is_on_same_value) * is_next_on_same_value
                + is_on_same_value * (1 - is_next_on_same_value)
            )

    return (
        grad_next_on_same_value,
        idx_on_same_value,
        is_next_on_same_value,
    )


@njit
def _backward_scan(
    value: np.ndarray,
    endog_grid: np.ndarray,
    exog_grid: np.ndarray,
    suboptimal_points: np.ndarray,
    jump_thresh: float,
    idx_current: int,
    idx_next: int,
) -> Tuple[float, int]:
    """Scan backward to check whether current point is optimal.

    Args:
        value (np.ndarray): 1d array containing the value function of shape
            (n_grid_wealth + 1,).
        endog_grid (np.ndarray): 1d array containing the endogenous wealth grid of
            shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid of
            shape (n_grid_wealth + 1,).
        suboptimal_points (list): List of suboptimal points in the value functions.
        jump_thresh (float): Threshold for the jump in the value function.
        idx_current (int): Index of the current point in the value function.
        idx_next (int): Index of the next point in the value function.

    Returns:
        tuple:

        - grad_before_on_same_value (float): The gradient of the previous point on
            the same value function.
        - is_before_on_same_value (int): Indicator for whether we have found a
            previous point on the same value function.

    """

    is_before_on_same_value = 0
    sub_idx_point_before_on_same_value = 0
    grad_before_on_same_value = 0

    indexes_reversed = len(suboptimal_points) - 1

    for i, idx_to_check in enumerate(suboptimal_points[::-1]):
        if endog_grid[idx_current] > endog_grid[idx_to_check]:
            is_on_same_value = (
                np.abs(
                    (exog_grid[idx_next] - exog_grid[idx_to_check])
                    / (endog_grid[idx_next] - endog_grid[idx_to_check])
                )
                < jump_thresh
            )
            is_before = is_on_same_value * (1 - is_before_on_same_value)
            sub_idx_point_before_on_same_value = (indexes_reversed - i) * is_before + (
                1 - is_before
            ) * sub_idx_point_before_on_same_value

            grad_before_on_same_value = (
                (value[idx_current] - value[idx_to_check])
                / (endog_grid[idx_current] - endog_grid[idx_to_check])
            ) * is_before + (1 - is_before) * grad_before_on_same_value

            is_before_on_same_value = (
                (is_before_on_same_value * is_on_same_value)
                + (1 - is_on_same_value) * is_before_on_same_value
                + is_on_same_value * (1 - is_before_on_same_value)
            )

    return (
        grad_before_on_same_value,
        sub_idx_point_before_on_same_value,
    )


@njit
def _evaluate_point_on_line(
    x1: float | np.ndarray,
    y1: float | np.ndarray,
    x2: float | np.ndarray,
    y2: float | np.ndarray,
    point_to_evaluate: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate a point on a line.

    Args:
        x1 (float): x coordinate of the first point.
        y1 (float): y coordinate of the first point.
        x2 (float): x coordinate of the second point.
        y2 (float): y coordinate of the second point.
        point_to_evaluate (float): The point to evaluate.

    Returns:
        float: The value of the point on the line.

    """
    return (y2 - y1) / (x2 - x1) * (point_to_evaluate - x1) + y1


@njit
def _linear_intersection(
    x1: float | np.ndarray,
    y1: float | np.ndarray,
    x2: float | np.ndarray,
    y2: float | np.ndarray,
    x3: float | np.ndarray,
    y3: float | np.ndarray,
    x4: float | np.ndarray,
    y4: float | np.ndarray,
) -> Tuple[float, float]:
    """Find the intersection of two lines.

    Args:

        x1 (float): x-coordinate of the first point of the first line.
        y1 (float): y-coordinate of the first point of the first line.
        x2 (float): x-coordinate of the second point of the first line.
        y2 (float): y-coordinate of the second point of the first line.
        x3 (float): x-coordinate of the first point of the second line.
        y3 (float): y-coordinate of the first point of the second line.
        x4 (float): x-coordinate of the second point of the second line.
        y4 (float): y-coordinate of the second point of the second line.

    Returns:
        tuple: x and y coordinates of the intersection point.

    """

    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y4 - y3) / (x4 - x3)

    x_intersection = (slope1 * x1 - slope2 * x3 + y3 - y1) / (slope1 - slope2)
    y_intersection = slope1 * (x_intersection - x1) + y1

    return x_intersection, y_intersection


@njit
def _append_index(x_array: np.ndarray, m: int):
    """Append a new point to an array."""
    for i in range(len(x_array) - 1):
        x_array[i] = x_array[i + 1]

    x_array[-1] = m
    return x_array


@njit
def _augment_grids(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    min_wealth_grid: float,
    n_constrained_points_to_add: int,
    value_function: Callable,
    value_function_args: Tuple,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extends the endogenous wealth grid, value, and policy functions to the left.

    Args:
        endog_grid (np.ndarray): 1d array containing the endogenous wealth grid of
            shape (n_endog_wealth_grid,), where n_endog_wealth_grid is of variable
            length depending on the number of kinks and non-concave regions in the
            value function.
        value (np.ndarray):  1d array storing the choice-specific
            value function of shape (n_endog_wealth_grid,), where
            n_endog_wealth_grid is of variable length depending on the number of
            kinks and non-concave regions in the value function.
            In the presence of kinks, the value function is a "correspondence"
            rather than a function due to non-concavities.
        policy (np.ndarray):  1d array storing the choice-specific
            policy function of shape (n_endog_wealth_grid,), where
            n_endog_wealth_grid is of variable length depending on the number of
            discontinuities in the policy function.
            In the presence of discontinuities, the policy function is a
            "correspondence" rather than a function due to multiple local optima.
        expected_value_zero_savings (float): The agent's expected value given that she
            saves zero.
        n_grid_wealth (int): Number of grid points in the exogenous wealth grid.

    Returns:
        tuple:

        - grid_augmented (np.ndarray): 1d array containing the augmented
            endogenous wealth grid with ancillary points added to the left.
        - policy_augmented (np.ndarray): 1d array containing the augmented
            policy function with ancillary points added to the left.
        - value_augmented (np.ndarray): 1d array containing the augmented
            value function with ancillary points added to the left.

    """
    grid_points_to_add = np.linspace(
        min_wealth_grid, endog_grid[0], n_constrained_points_to_add + 1
    )[:-1]

    values_to_add = np.empty_like(grid_points_to_add)

    for i, grid_point in enumerate(grid_points_to_add):
        values_to_add[i] = value_function(grid_point, *value_function_args)

    grid_augmented = np.append(grid_points_to_add, endog_grid)
    value_augmented = np.append(values_to_add, value)
    policy_augmented = np.append(grid_points_to_add, policy)

    return grid_augmented, value_augmented, policy_augmented


@njit
def _initialize_refined_arrays(
    value: np.ndarray, policy: np.ndarray, endog_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    value_refined = np.empty_like(value)
    policy_refined = np.empty_like(policy)
    endog_grid_refined = np.empty_like(endog_grid)

    value_refined[:] = np.nan
    policy_refined[:] = np.nan
    endog_grid_refined[:] = np.nan

    value_refined[:2] = value[:2]
    policy_refined[:2] = policy[:2]
    endog_grid_refined[:2] = endog_grid[:2]

    return value_refined, policy_refined, endog_grid_refined
