"""Interactive plotting dashboard for housing renting model analysis.

This module provides an interactive dashboard using Plotly for analyzing
EGM plots with zoom, pan, and hover capabilities. No need to rerun the solver
to adjust plot axes - just interact with the plots directly!

Main Features
-------------
- Interactive zoom and pan on all plots
- Hover tooltips showing exact values
- Toggle visibility of different series
- Export plots as static images
- Side-by-side comparison of methods
- Synchronized axes for easy comparison

Usage
-----
After running the solver:
    from helpers.interactive_plots import create_egm_dashboard
    create_egm_dashboard(model, methods, output_dir)
    
Or load from saved solutions:
    create_egm_dashboard(None, methods, output_dir, load_dir="outputs/solutions")
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dynx.stagecraft.solmaker import Solution


def create_egm_dashboard(model, methods, output_dir, plot_period=0, 
                         load_dir=None, H_indices=None, y_idx_list=None,
                         save_html=True, auto_open=True):
    """
    Create an interactive dashboard for EGM plots with full zoom/pan capabilities.
    
    Parameters
    ----------
    model : ModelCircuit or None
        Solved model circuit, or None if loading from disk
    methods : list of str
        List of methods to plot (e.g., ['FUES', 'DCEGM', 'VFI_HDGRID'])
    output_dir : str
        Directory to save the HTML dashboard
    plot_period : int
        Period to plot
    load_dir : str, optional
        Directory to load solutions from if model is None
    H_indices : list of int, optional
        Housing indices to plot. If None, uses [0, mid, max]
    y_idx_list : list of int, optional
        Income indices to plot. If None, uses [0]
    save_html : bool
        Whether to save the dashboard as an HTML file
    auto_open : bool
        Whether to automatically open the dashboard in browser
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive figure object
    """
    
    if y_idx_list is None:
        y_idx_list = [0]
    
    # Create a color palette for methods
    colors = px.colors.qualitative.Set1
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    
    # Create subplot structure - one row per y_idx, two columns (value and assets)
    n_rows = len(y_idx_list)
    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=[f"Value Function (y_idx={y})" for y in y_idx_list] + 
                      [f"Assets (y_idx={y})" for y in y_idx_list],
        horizontal_spacing=0.12,
        vertical_spacing=0.15 if n_rows > 1 else 0.2
    )
    
    # Track all data for each method
    all_data = {}
    
    for method in methods:
        method_data = load_egm_data(model, method, plot_period, load_dir)
        if method_data is None:
            print(f"Warning: No data found for method {method}")
            continue
            
        all_data[method] = method_data
        
        # Determine H_indices if not specified
        if H_indices is None and 'H_grid' in method_data:
            H_grid = method_data['H_grid']
            H_indices = [0, len(H_grid) // 2, len(H_grid) - 1]
        
        # Plot for each y_idx
        for row_idx, y_idx in enumerate(y_idx_list):
            row = row_idx + 1
            
            # Plot for each H_idx
            for H_idx in H_indices if H_indices else [0]:
                grid_key = f"{y_idx}-{H_idx}"
                
                # Get data for this combination
                plot_data = extract_grid_data(method_data, grid_key, H_idx)
                if plot_data is None:
                    continue
                
                # Determine H value for label
                h_val = method_data['H_grid'][H_idx] if 'H_grid' in method_data else H_idx
                
                # Plot value function (refined)
                if 'e_refined' in plot_data and 'vf_refined' in plot_data:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data['e_refined'],
                            y=plot_data['vf_refined'],
                            mode='lines+markers',
                            name=f"{method} H={h_val:.2f}",
                            line=dict(color=method_colors[method], width=2),
                            marker=dict(size=4),
                            legendgroup=f"{method}_{H_idx}",
                            hovertemplate='<b>%{fullData.name}</b><br>e: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>'
                        ),
                        row=row, col=1
                    )
                
                # Plot unrefined points as scatter
                if 'e_unrefined' in plot_data and 'vf_unrefined' in plot_data:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data['e_unrefined'],
                            y=plot_data['vf_unrefined'],
                            mode='markers',
                            name=f"{method} H={h_val:.2f} (unrefined)",
                            marker=dict(
                                color='rgba(0,0,0,0)',
                                size=8,
                                line=dict(color=method_colors[method], width=1)
                            ),
                            legendgroup=f"{method}_{H_idx}",
                            showlegend=False,
                            hovertemplate='<b>Unrefined</b><br>e: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>'
                        ),
                        row=row, col=1
                    )
                
                # Plot assets (refined)
                if 'e_refined' in plot_data and 'a_refined' in plot_data:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data['e_refined'],
                            y=plot_data['a_refined'],
                            mode='lines+markers',
                            name=f"{method} H={h_val:.2f}",
                            line=dict(color=method_colors[method], width=2),
                            marker=dict(size=4),
                            legendgroup=f"{method}_{H_idx}",
                            showlegend=False,
                            hovertemplate='<b>%{fullData.name}</b><br>e: %{x:.3f}<br>Assets: %{y:.3f}<extra></extra>'
                        ),
                        row=row, col=2
                    )
                
                # Plot unrefined assets as scatter
                if 'e_unrefined' in plot_data and 'a_unrefined' in plot_data:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data['e_unrefined'],
                            y=plot_data['a_unrefined'],
                            mode='markers',
                            name=f"{method} H={h_val:.2f} (unrefined)",
                            marker=dict(
                                color='rgba(0,0,0,0)',
                                size=8,
                                line=dict(color=method_colors[method], width=1)
                            ),
                            legendgroup=f"{method}_{H_idx}",
                            showlegend=False,
                            hovertemplate='<b>Unrefined</b><br>e: %{x:.3f}<br>Assets: %{y:.3f}<extra></extra>'
                        ),
                        row=row, col=2
                    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Interactive EGM Dashboard - Period {plot_period}",
            font=dict(size=20)
        ),
        height=400 * n_rows + 100,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        template='plotly_white'
    )
    
    # Update axes labels
    for row_idx in range(n_rows):
        row = row_idx + 1
        fig.update_xaxes(title_text="Endogenous State (e)", row=row, col=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Endogenous State (e)", row=row, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Value", row=row, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Assets", row=row, col=2, gridcolor='lightgray')
    
    # Save HTML if requested
    if save_html:
        html_path = os.path.join(output_dir, "egm_interactive_dashboard.html")
        fig.write_html(html_path)
        print(f"Interactive dashboard saved to: {html_path}")
        print("Open this file in your browser to interact with the plots!")
    
    # Show the figure if requested
    if auto_open:
        fig.show()
    
    return fig


def create_policy_dashboard(model, methods, output_dir, plot_period=0,
                           load_dir=None, save_html=True, auto_open=True):
    """
    Create an interactive dashboard for policy function plots.
    
    Similar to create_egm_dashboard but for consumption, housing, and tenure policies.
    """
    
    # Create subplot structure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Consumption Policy", "Housing Policy", 
                       "Value Function", "Q Function"],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    colors = px.colors.qualitative.Set2
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    
    for method in methods:
        # Load solution
        sol = load_solution(model, method, plot_period, load_dir, stage="OWNC")
        if sol is None:
            continue
        
        # Get grids (assuming we saved them or can reconstruct)
        # This is a simplified version - you'd need to adapt based on your actual data structure
        
        # Plot consumption policy
        if hasattr(sol, 'policy') and hasattr(sol.policy, 'c'):
            c_policy = sol.policy.c
            # Assuming we have w_grid somewhere
            w_grid = np.linspace(0, 50, c_policy.shape[0])  # Placeholder
            
            # Plot for different H values
            for H_idx in [0, c_policy.shape[1]//2, c_policy.shape[1]-1]:
                fig.add_trace(
                    go.Scatter(
                        x=w_grid,
                        y=c_policy[:, H_idx, 0],  # y_idx=0
                        mode='lines',
                        name=f"{method} H_idx={H_idx}",
                        line=dict(color=method_colors[method]),
                        legendgroup=method,
                        hovertemplate='<b>%{fullData.name}</b><br>w: %{x:.2f}<br>c: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
    
    # Update layout
    fig.update_layout(
        title="Interactive Policy Function Dashboard",
        height=800,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Cash-on-Hand (w)", gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    # Save HTML if requested
    if save_html:
        html_path = os.path.join(output_dir, "policy_interactive_dashboard.html")
        fig.write_html(html_path)
        print(f"Interactive policy dashboard saved to: {html_path}")
    
    if auto_open:
        fig.show()
    
    return fig


def load_egm_data(model, method, plot_period, load_dir=None):
    """Load EGM data for a specific method."""
    
    # Load solution
    sol = load_solution(model, method, plot_period, load_dir, stage="OWNC")
    if sol is None:
        return None
    
    # Extract EGM data
    data = {}
    
    # Try to get EGM structure
    if hasattr(sol, 'EGM'):
        data['egm'] = sol.EGM
    elif hasattr(sol, '_arr') and 'EGM' in sol._arr:
        data['egm'] = sol._arr['EGM']
    else:
        return None
    
    # Try to get grid information (might need to load separately or reconstruct)
    # This is a placeholder - you'd need to adapt based on your actual setup
    if model is not None:
        period = model.get_period(plot_period)
        stage = period.get_stage("OWNC")
        if stage and stage.dcsn and stage.dcsn.grid:
            data['H_grid'] = stage.dcsn.grid.H_nxt
            data['w_grid'] = stage.dcsn.grid.w
    
    return data


def load_solution(model, method, plot_period, load_dir=None, stage="OWNC"):
    """Load a solution either from model or from disk."""
    
    if model is not None:
        # Get from live model
        period = model.get_period(plot_period)
        stage_obj = period.get_stage(stage)
        if stage_obj and stage_obj.dcsn:
            return stage_obj.dcsn.sol
    elif load_dir is not None:
        # Load from disk
        fname = f"{load_dir}/{method}/{stage}_dcsn_period{plot_period}"
        if os.path.exists(f"{fname}.npz"):
            return Solution.load(fname)
    
    return None


def extract_grid_data(method_data, grid_key, H_idx):
    """Extract grid data for a specific y-H combination."""
    
    if 'egm' not in method_data:
        return None
    
    egm_struct = method_data['egm']
    data = {}
    
    # Get unrefined data
    if 'unrefined' in egm_struct:
        unrefined = egm_struct['unrefined']
        prefixed_key = f"e_{grid_key}"
        if prefixed_key in unrefined:
            data['e_unrefined'] = unrefined[prefixed_key]
            data['vf_unrefined'] = unrefined.get(f"Q_{grid_key}")
            data['c_unrefined'] = unrefined.get(f"c_{grid_key}")
            data['a_unrefined'] = unrefined.get(f"a_{grid_key}")
    
    # Get refined data
    if 'refined' in egm_struct:
        refined = egm_struct['refined']
        prefixed_key = f"e_{grid_key}"
        if prefixed_key in refined:
            data['e_refined'] = refined[prefixed_key]
            data['vf_refined'] = refined.get(f"Q_{grid_key}")
            data['c_refined'] = refined.get(f"c_{grid_key}")
            data['a_refined'] = refined.get(f"a_{grid_key}")
    
    # Filter out None values
    data = {k: v for k, v in data.items() if v is not None}
    
    return data if data else None


def compare_methods_interactive(model, methods, output_dir, plot_period=0,
                               load_dir=None, baseline_method=None):
    """
    Create an interactive comparison dashboard showing differences between methods.
    
    Parameters
    ----------
    model : ModelCircuit or None
        Solved model circuit
    methods : list of str
        Methods to compare
    output_dir : str
        Output directory
    plot_period : int
        Period to analyze
    load_dir : str, optional
        Directory to load solutions from
    baseline_method : str, optional
        Method to use as baseline for comparison. If None, uses first method.
    """
    
    if baseline_method is None:
        baseline_method = methods[0]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Consumption Difference", "Value Difference",
                       "Consumption (Absolute)", "Value (Absolute)"],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    # Load baseline
    baseline_sol = load_solution(model, baseline_method, plot_period, load_dir)
    if baseline_sol is None:
        print(f"Could not load baseline method {baseline_method}")
        return None
    
    colors = px.colors.qualitative.Set3
    
    for i, method in enumerate(methods):
        if method == baseline_method:
            continue
            
        sol = load_solution(model, method, plot_period, load_dir)
        if sol is None:
            continue
        
        # Compare consumption policies
        if hasattr(sol, 'policy') and hasattr(baseline_sol, 'policy'):
            c_diff = sol.policy.c - baseline_sol.policy.c
            
            # Create heatmap of differences
            fig.add_trace(
                go.Heatmap(
                    z=c_diff[:, :, 0],  # y_idx=0
                    colorscale='RdBu',
                    zmid=0,
                    name=f"{method} vs {baseline_method}",
                    hovertemplate='w_idx: %{x}<br>H_idx: %{y}<br>Diff: %{z:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
    
    fig.update_layout(
        title=f"Method Comparison Dashboard (Baseline: {baseline_method})",
        height=800,
        template='plotly_white'
    )
    
    # Save
    html_path = os.path.join(output_dir, "method_comparison_dashboard.html")
    fig.write_html(html_path)
    print(f"Comparison dashboard saved to: {html_path}")
    
    return fig