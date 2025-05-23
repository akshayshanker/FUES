""" Script to plot solution of Ishkakov et al (2017) retirement choice
model using FUES-EGM by Dobrescu and Shanker (2024).

Author: Akshay Shanker, University of New South Wales, akshay.shanker@me.com.

See examples/retirement_choice for example module. 

Todo
----
- Improve integration with DC-EGM and implement
    timing comparison with DC-EGM with jit compiled
    version of DC-EGM


"""

import numpy as np
import time
from HARK.interpolation import LinearInterp
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl
import matplotlib.lines as mlines


# Import local modules
import os,sys 
# Ensure project root is on sys.path so that `import examples` works when this
# script is executed from the *scripts/* directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
#cwd = os.getcwd()
#sys.path.append('..')
#os.chdir(cwd)
from dc_smm.fues.fues import FUES as fues_alg
from dc_smm.models.retirement.retirement import Operator_Factory, RetirementModel, euler

def plot_egrids(age, e_grid, vf_work, c_worker, del_a, g_size, cp, tag = 'sigma0'):
    """ 

    Plot unrefined vs refined endogenous grid for age. 
    Left plot is value, right plot is policy points 
    Figure 4. in FUES version Oct 2024. 
    
    Parameters
    ----------
    age : int
        Age at which to plot value function and policy
    e_grid : dict
        Dictionary of endogenous grids for worker
    vf_work : dict
        Dictionary of value unrefined corrs. for worker by age
    c_worker : dict
        Dictionary of unrefined consumption points for worker by age
    del_a : dict
        Dictionary of unrefined derivative of policy function for worker by age
    g_size : int
        Grid size for the model for labeling.
    cp : RetirementModel
        RetirementModel instance
    tag : str, optional
        Tag for labeling, default is 'sigma0'.
     
    Returns 
    -------
    None 


    """ 

    # 1. Get unrefined endogenous grid, value function and consumption
    # for worker at time t
    x = np.array(e_grid[age])
    vf = np.array(vf_work[age])
    c = np.array(c_worker[age])
    del_a = np.array(del_a[age])
    a_prime = np.array(cp.asset_grid_A)

    # 2. Generate refined grid, value function and policy using FUES
    x_clean, vf_clean, c_clean, a_prime_clean, del_a_clean \
        = fues_alg(x, vf, c, a_prime,del_a, 0.8)

    # 3. make plots  left 
    pl.close()
    fig, ax = pl.subplots(1, 2)
    sns.set(
        style="white", rc={
            "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

    ax[0].scatter(
        x,
        vf * cp.beta - cp.delta,
        s=20,
        facecolors='none',
        edgecolors='r', label='EGM points')
    
    ax[0].scatter(
        x_clean,
        vf_clean * cp.beta - cp.delta,
        color='blue',
        s=15,
        marker='x',
        linewidth=0.75, label='FUES optimal points')
    ax[0].plot(
        x_clean,
        vf_clean * cp.beta - cp.delta,
        color='black',
        linewidth=1,
        label=r'Value function $V_t^{1}$')

    # formatting     
    ax[0].set_ylabel('Value', fontsize=11)
    ax[0].set_ylim(7.6, 8.4001)
    ax[0].set_xlim(44, 55.01)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].set_yticks(ax[0].get_yticks())
    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].tick_params(axis='y', labelsize=9)
    ax[0].tick_params(axis='x', labelsize=9)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].grid(True)

    # right plot 
    ax[1].scatter(
        np.sort(x),
        np.take(
            x - c,
            np.argsort(x)),
        s=20,
        facecolors='none',
        edgecolors='r',
        label='EGM points')
    ax[1].scatter(
        np.sort(x_clean),
        np.take(
            x_clean - c_clean,
            np.argsort(x_clean)),
        s=20,
        color='blue',
        marker='x',
        linewidth=0.75,
        label='FUES optimal points')
    
    # fromatting 
    ax[1].set_ylim(20, 40)
    ax[1].set_xlim(44, 55.01)
    ax[1].set_ylabel('Financial assets at time t+1', fontsize=11)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_yticks(ax[1].get_yticks())
    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].tick_params(axis='y', labelsize=9)
    ax[1].tick_params(axis='x', labelsize=9)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax[1].legend(frameon=False, prop={'size': 10})
    ax[1].grid(True)

    # common x label
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.supxlabel('Financial assets at time t', fontsize=11)
    fig.savefig(
        'results/plots/retirement/ret_vf_aprime_all_{}_{}_{}.png'.format(age, g_size, tag))
    pl.close()

    return None

def plot_cons_pol(sigma_work, cp, ages = [17,10,0]):

    """
    Plot consumption policy for difference ages.

    Parameters
    ----------
    sigma_work : dict
        Dictionary of consumption policy functions by age
    cp : RetirementModel
        RetirementModel instance
    ages : list
        List of ages to plot consumption policy for
    
    Returns
    -------
    None

    """
    
    # Plot consumption policy  for difference ages
    sns.set(style="whitegrid",
            rc={"font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 10})
    fig, ax = pl.subplots(1, 1)

    for t, col, lab in zip(ages, ['blue', 'red', 'black'], [
            't=18', 't=10', 't=1']):

        cons_pol = np.copy(sigma_work[t])

        # remove jump joints for plotting only
        pos = np.where(np.abs(np.diff(cons_pol))\
                    /np.diff(cp.asset_grid_A)> 0.3)[0] + 1
        y1 = np.insert(cons_pol, pos, np.nan)
        x1 = np.insert(cp.asset_grid_A, pos, np.nan)

        ax.plot(x1, y1, color=col, label=lab)
        ax.set_xlim(0, 380)
        ax.set_ylim(0, 40)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.set_ylabel('Consumption at time $t$', fontsize=11)
        ax.set_xlabel('Financial assets at time $t$', fontsize=11)

    ax.legend(frameon=False, prop={'size': 10})
    fig.savefig('results/plots/retirement/ret_cons_all.png'.format(t))
    pl.close()

    return None

def plot_dcegm_cf(
    age, g_size, e_grid, vf_work, c_worker, dela_worker, a_prime, cp, 
    tag='sigma05', plot=True
):
    """
    Figure 5 in FUES version Oct 2024.

    Plot to compare DC-EGM and FUES for worker at a specific age.

    Parameters
    ----------
    age : int
        Age at which to plot value function and policy.
    g_size : int
        Grid size for the model for labeling.
    e_grid : dict
        Dictionary of endogenous grids for worker.
    vf_work : dict
        Dictionary of value unrefined correlations for worker by age.
    c_worker : dict
        Dictionary of unrefined consumption points for worker by age.
    dela_worker : dict
        Dictionary of unrefined derivative of policy function by age.
    a_prime : str
        Taste shock (sigma) for labeling.
    cp : RetirementModel
        RetirementModel instance
    tag : str, optional
        Tag for labeling, default is 'sigma05'.
    plot : bool, optional
        If True, generates plot. Default is True.
    """
    x = e_grid[age]
    vf = vf_work[age]
    c = c_worker[age]
    a_prime = cp.asset_grid_A
    dela = dela_worker[age]
    time_start_dcegm = time.time()

    x_clean, vf_clean, c_clean, a_prime_clean, dela_clean = fues_alg(
        x, vf, c, a_prime, dela, m_bar=1, endog_mbar=True
    )

    vf_interp_fues = np.interp(x, x_clean, vf_clean)
    vf_interp_fues[x.searchsorted(x_clean)] = vf_clean

    start, end = calc_nondecreasing_segments(x, vf)
    segments, c_segments, a_segments, m_segments = [], [], [], []
    v_segments, dela_segments = [], []

    for j in range(len(start)):
        idx = range(start[j], end[j] + 1)
        segments.append([x[idx], vf[idx]])
        c_segments.append(c[idx])
        a_segments.append(a_prime[idx])
        m_segments.append(x[idx])
        v_segments.append(vf[idx])
        dela_segments.append(dela[idx])

    m_upper, v_upper, inds_upper = upper_envelope(segments, calc_crossings=False)
    c1_env = np.zeros_like(m_upper) + np.nan
    a1_env = np.zeros_like(m_upper) + np.nan
    v1_env = np.zeros_like(m_upper) + np.nan
    d1_env = np.zeros_like(m_upper) + np.nan

    for k, c_segm in enumerate(c_segments):
        c1_env[inds_upper == k] = c_segm[m_segments[k].searchsorted(
            m_upper[inds_upper == k])]

    for k, a_segm in enumerate(a_segments):
        a1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k],
                                            m_segments[k], a_segm)
    for k, v_segm in enumerate(v_segments):
        v1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k], m_segments[k], v_segm)
    
    for k, dela_segm in enumerate(dela_segments):
        d1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k], m_segments[k], dela_segm)

    a1_up = LinearInterp(m_upper, a1_env)
    indices = np.where(np.in1d(a1_env, a_prime))[0]
    a1_env2 = a1_env[indices]
    m_upper2 = m_upper[indices]
    c_env2 = c1_env[indices]
    v_env2 = v1_env[indices]
    d_env2 = d1_env[indices]

    # Plotting 
    if plot:

        pl.close()
        fig, ax = pl.subplots(1, 2)
        sns.set(
            style="whitegrid", rc={
                "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

        ax[0].scatter(
            x,
            vf * cp.beta - cp.delta,
            s=20,
            facecolors='none',
            edgecolors='r', label='EGM points')
        ax[0].scatter(
            x_clean,
            vf_clean * cp.beta - cp.delta,
            color='blue',
            s=15,
            marker='x',
            linewidth=0.75,
            label='FUES optimal points')

        ax[1].scatter(
            x,
            a_prime,
            edgecolors='r',
            s=15,
            facecolors='none',
            label='EGM points',
            linewidth=0.75)

        for k, v_segm in enumerate(v_segments):
            x_values = m_segments[k]
            y_values = v_segm * cp.beta - cp.delta
            
            # Check if it's a single point or a segment
            if len(x_values) == 1:
                # If it's a single point, just scatter it with a marker
                ax[1].scatter(x_values, y_values, color='black', marker='x', linewidth=0.75)
            else:
                # Plot the line segment with markers at the ends
                ax[1].plot(x_values, y_values, color='black', linewidth=0.75)
                
                # Add markers at the start and end of the line
                ax[1].scatter([x_values[0], x_values[-1]], [y_values[0], y_values[-1]], 
                            color='black', marker='x', linewidth=0.75)

        ax[1].scatter(
            x_clean,
            a_prime_clean,
            color='blue',
            s=15,
            marker='x',
            label='FUES optimal points',
            linewidth=0.75)

        # Plot the line segments and scatter the points at ends
        for k, v_segm in enumerate(v_segments):
            x_values = m_segments[k]
            y_values = v_segm * cp.beta - cp.delta
            if len(x_values) == 1:
                ax[0].scatter(x_values, y_values, color='black', marker='x', linewidth=0.75)
            else:
                ax[0].plot(x_values, y_values, color='black', linewidth=0.75)
                ax[0].scatter([x_values[0], x_values[-1]], [y_values[0], y_values[-1]],\
                                color='black', marker='x', linewidth=0.75)


        for k, a_segm in enumerate(a_segments):
            x_values = m_segments[k]
            y_values = a_segm
            if len(x_values) == 1:
                ax[1].scatter(x_values, y_values, color='black', marker='x', linewidth=0.75)
            else:
                ax[1].plot(x_values, y_values, color='black', linewidth=0.75)
                ax[1].scatter([x_values[0], x_values[-1]], [y_values[0],\
                     y_values[-1]], color='black', marker='x', linewidth=0.75)

        # Collect the automatic legend handles and labels for ax[0]
        handles0, labels0 = ax[0].get_legend_handles_labels()

        # Append the custom '-x' line to the existing handles and labels
        line_x_end_handle = mlines.Line2D([0, 1], [0, 0], color='black',\
                                                linestyle='-', linewidth=0.75)  
        marker_start_handle = mlines.Line2D([0], [0], color='black', marker='x',\
                                                 linestyle='None', markersize=6)  
        marker_end_handle = mlines.Line2D([0], [0], color='black', marker='x',\
                                                linestyle='None', markersize=6)  

        # Collect the automatic legend handles and labels for ax[0]
        handles0, labels0 = ax[1].get_legend_handles_labels()

        # Append the custom '-x' line to the existing handles and labels
        handles0.append((line_x_end_handle, marker_start_handle, marker_end_handle))
        labels0.append('DC-EGM segments')

        # Set lims
        ax[0].set_ylim(7.6, 8.4001)
        ax[0].set_xlim(44, 55.1)
        ax[1].set_ylim(20, 40)
        ax[1].set_xlim(44, 55.1)
        
        # reformat labels for ticks
        ax[0].set_yticks(ax[0].get_yticks())
        ax[0].set_xticks(ax[0].get_xticks())
        ax[0].tick_params(axis='y', labelsize=9)
        ax[0].tick_params(axis='x', labelsize=9)

        ax[1].set_yticks(ax[1].get_yticks())
        ax[1].set_xticks(ax[1].get_xticks())
        ax[1].tick_params(axis='y', labelsize=9)
        ax[1].tick_params(axis='x', labelsize=9)

        # format for ticks decimal 
        ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        # Y- Axis labels
        ax[0].set_ylabel('Value', fontsize=11)
        ax[1].set_ylabel('Financial assets at time t+1', fontsize=11)
        
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[0].spines['left'].set_visible(True)
        ax[0].spines['bottom'].set_visible(True)
        
        # add legends to both
        ax[0].legend(frameon=False, prop={'size': 10})
        ax[1].legend(frameon=False, prop={'size': 10})
        ax[1].legend(handles=handles0, labels=labels0, frameon=False,\
                                     prop={'size': 10}, loc='upper left'
        )
        ax[0].legend(handles=handles0, labels=labels0, frameon=False,\
                                     prop={'size': 10}, loc='upper left'
        )
        

        # add common x label
        fig.supxlabel('Financial assets at time t', fontsize=11)
        fig.tight_layout()
        fig.savefig('results/plots/retirement/ret_vf_aprime_all_{}_cf_{}_{}.png'
                    .format(g_size, age,tag))

    return v_upper, v_env2, vf_clean, a_prime_clean, m_upper2, a1_env2

def test_Timings(grid_sizes, delta_values,n =3):
    # Initialize lists to hold results for LaTeX tables
    latex_errors_data = []
    latex_timings_data = []

    for g_size_baseline in grid_sizes:
        for delta in delta_values:
            print(f"\nTesting with grid size: {g_size_baseline}\
                        and delta: {delta}"\
            )

            # Create instance of RetirementModel
            cp = RetirementModel(
                r=0.02,
                beta=0.96,
                delta=delta,
                y=20,
                b=1E-100,
                grid_max_A=500,
                grid_size=g_size_baseline,
                T=50,
                smooth_sigma=0,
                padding_mbar= -0.011,
            )

            # Unpack solver operators 
            Ts_ret, Ts_work, iter_bell = Operator_Factory(cp)

            # Initialize variables to store best times and errors
            best_time_RFC = float('inf')
            best_time_FUES = float('inf')
            best_time_DCEGM = float('inf')
            best_time_CONSAV = float('inf')
            best_error_RFC = float('inf')
            best_error_FUES = float('inf')
            best_error_DCEGM = float('inf')
            best_error_CONSAV = float('inf')

            for _ in range(n):  # Run each test n times and take the best
                # Test RFC
                _, _, _, _, c_refined_RFC, _, iter_time_age = iter_bell(
                    cp, method='RFC'
                )
                time_end_RFC = np.mean(iter_time_age[0])
                Euler_error_RFC = euler(cp, c_refined_RFC)

                # Test FUES
                _, _, _, _, c_refined_FUES, _, iter_time_age = iter_bell(
                    cp, method='FUES'
                )
                time_end_FUES = np.mean(iter_time_age[0])
                Euler_error_FUES = euler(cp, c_refined_FUES)

                # Test DCEGM
                _, _, _, _, c_refined_DCEGM, _, iter_time_age = iter_bell(
                    cp, method='DCEGM'
                )
                time_end_DCEGM = np.mean(iter_time_age[0])
                Euler_error_DCEGM = euler(cp, c_refined_DCEGM)
                
                # Test CONSAV
                _, _, _, _, c_refined_CONSAV, _, iter_time_age = iter_bell(
                    cp, method='CONSAV'
                )
                time_end_CONSAV = np.mean(iter_time_age[0])
                Euler_error_CONSAV = euler(cp, c_refined_CONSAV)

                # Take the best of 3 runs for timings
                best_time_RFC = min(best_time_RFC, time_end_RFC)
                best_time_FUES = min(best_time_FUES, time_end_FUES)
                best_time_DCEGM = min(best_time_DCEGM, time_end_DCEGM)
                best_time_CONSAV = min(best_time_CONSAV, time_end_CONSAV)

                # Take the best of 3 runs for errors
                best_error_RFC = min(best_error_RFC, Euler_error_RFC)
                best_error_FUES = min(best_error_FUES, Euler_error_FUES)
                best_error_DCEGM = min(best_error_DCEGM, Euler_error_DCEGM)
                best_error_CONSAV = min(best_error_CONSAV, Euler_error_CONSAV)

            # Store the best results for the LaTeX tables
            latex_errors_data.append([
                g_size_baseline, delta, best_error_RFC,
                best_error_FUES, best_error_DCEGM, best_error_CONSAV
            ])
            latex_timings_data.append([
                g_size_baseline, delta, best_time_RFC * 1000,
                best_time_FUES * 1000, best_time_DCEGM * 1000, best_time_CONSAV * 1000
            ])

            # Print results for current grid size and delta
            print(
                f'Euler errors for grid size {g_size_baseline}, delta {delta}: '
                f'RFC: {best_error_RFC:.6f}, FUES: {best_error_FUES:.6f}, '
                f'DCEGM: {best_error_DCEGM:.6f}, CONSAV: {best_error_CONSAV:.6f}'
            )
            print(
                f'Timings for grid size {g_size_baseline}, delta {delta}: '
                f'RFC: {best_time_RFC:.6f}, FUES: {best_time_FUES:.6f}, '
                f'DCEGM: {best_time_DCEGM:.6f}, CONSAV: {best_time_CONSAV:.6f}'
            )

    # Generate LaTeX tables 
    generate_latex_table(latex_timings_data,latex_errors_data,\
                                "timing", "Retirement model"
    )

def generate_latex_table(data, errors, table_type, caption):
    """
    Generates a LaTeX table with performance (RFC, FUES, DCEGM, CONSAV) and Euler errors.

    Parameters:
    data : list of lists
        Data for RFC, FUES, DCEGM, CONSAV timings.
    errors : list of lists
        Data for corresponding Euler errors.
    table_type : str
        Type of the table for labeling.
    caption : str
        Caption for the LaTeX table.
    """
    # Header for LaTeX table with multirow and formatting
    latex_code = f"""
        \\begin{{table}}[htbp]
        \\centering
        \\small
        \\begin{{tabular}}{{cccccc|cccc}}
        \\toprule
        \\multirow{{2}}{{*}}{{\\textit{{Grid Size}}}} & 
        \\multirow{{2}}{{*}}{{\\textit{{Delta}}}} & 
        \\multicolumn{{4}}{{c}}{{\\textbf{{Timing (Seconds)}}}} & 
        \\multicolumn{{4}}{{c}}{{\\textbf{{Euler Error (Log10)}}}} \\\\
        & & \\textbf{{RFC}} & \\textbf{{FUES}} & \\textbf{{DCEGM}} & \\textbf{{CONSAV}} & 
        \\textbf{{RFC}} & \\textbf{{FUES}} & \\textbf{{DCEGM}} & \\textbf{{CONSAV}} \\\\
        \\midrule
        """

    unique_grid_sizes = np.unique([row[0] for row in data])

    for grid_size in unique_grid_sizes:
        filtered_data = [row for row in data if row[0] == grid_size]
        filtered_errors = [row for row in errors if row[0] == grid_size]

        if len(filtered_data) != len(filtered_errors):
            raise ValueError(
                f"Mismatch in data and errors for grid size {grid_size}. "
                f"Check the input data."
            )

        latex_code += f"\\multirow{{{len(filtered_data)}}}{{*}}{{\\textit{{"
        latex_code += f"{int(grid_size)}}}}} "

        for i, row in enumerate(filtered_data):
            if i < len(filtered_errors):
                error_row = filtered_errors[i]
                
                # Explicitly map data indices to ensure correct order
                delta = row[1]
                rfc_time = row[2]
                fues_time = row[3]
                dcegm_time = row[4]
                consav_time = row[5]
                
                rfc_error = error_row[2]
                fues_error = error_row[3]
                dcegm_error = error_row[4]
                consav_error = error_row[5]
                
                row_str = (
                    f"& {delta:.2f} & {rfc_time:.3f} & {fues_time:.3f} & "
                    f"{dcegm_time:.3f} & {consav_time:.3f} & {rfc_error:.3f} & "
                    f"{fues_error:.3f} & {dcegm_error:.3f} & {consav_error:.3f} \\\\\n"
                )
                
                if i == 0:
                    latex_code += row_str
                else:
                    latex_code += f" {row_str}"
            else:
                raise IndexError(
                    "Mismatch: filtered_errors has fewer elements than "
                    f"filtered_data for grid size {grid_size}."
                )

        latex_code += "\\midrule\n"

    latex_code += f"""
        \\bottomrule
        \\end{{tabular}}
        \\caption{{\\textit{{{caption}}} comparison across different grid sizes, 
        delta values, and accuracy (Euler error)}}
        \\label{{tab:{table_type}_comparison}}
        \\end{{table}}
        """

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)

    file_path = os.path.join(results_dir, f"retirement_{table_type}.tex")

    with open(file_path, "w") as file:
        file.write(latex_code)


if __name__ == "__main__":

    
    grid_sizes = [500, 1000, 2000, 3000]  # Test different grid sizes
    delta_values = [0.25, 0.5, 1, 1.5, 2]  # Test different delta values
    egrid_plot_age = 17
    run_performance_tests = True

    # Test performance of different methods for RetirementModel across grid sizes
    if run_performance_tests:
        test_Timings(grid_sizes, delta_values)

    # Generate baseline solution using FUES and make plots
    g_size_baseline = 3000

    cp = RetirementModel(
        r=0.02, beta=0.98, delta=1, y=20, b=1E-10, grid_max_A=500,
        grid_size=3000, T=20, smooth_sigma=0
    )

    cp2 = RetirementModel(
        r=0.02, beta=0.98, delta=1, y=20, b=1E-10, grid_max_A=500,
        grid_size=3000, T=20, smooth_sigma=0
    )

    # Unpack solver operators
    Ts_ret, Ts_work, iter_bell = Operator_Factory(cp)
    Ts_ret, Ts_work, iter_bell2 = Operator_Factory(cp2)

    # Get optimal value and policy functions using FUES by iterating on Bellman
    # precompile numba functions
    _ = iter_bell(cp, method='RFC')
    e_grid_worker_unref, vf_work_unref, vf_refined, c_worker_unref, \
        c_refined_RFC, dela_unrefined, time_end_RFC = iter_bell(
            cp, method='RFC'
        )

    # precompile numba functions
    _ = iter_bell(cp, method='FUES')
    _, _, _, _, c_refined_FUES, _, time_end_FUES = iter_bell(cp, method='FUES')

    # precompile numba functions
    _ = iter_bell(cp, method='DCEGM')
    _, _, _, _, c_refined_DCEGM, _, time_end_DCEGM = iter_bell(cp, method='DCEGM')

    # precompile numba functions
    _ = iter_bell(cp, method='CONSAV')
    _, _, _, _, c_refined_CONSAV, _, time_end_CONSAV = iter_bell(cp, method='CONSAV')

    Euler_error_RFC = euler(cp, c_refined_RFC)
    Euler_error_FUES = euler(cp, c_refined_FUES)
    Euler_error_DCEGM = euler(cp, c_refined_DCEGM)
    Euler_error_CONSAV = euler(cp, c_refined_CONSAV)
    print(
        "| Method | Euler Error    | Avg. upper env. time(ms) | Total solution time(ms) |\n"
        "|--------|----------------|--------------------------|-------------------------|\n"
        f"| RFC    | {Euler_error_RFC: <14.6f} | {time_end_RFC[0]*1000: <24.6f} | {time_end_RFC[1]*1000: <23.6f} |\n"
        f"| FUES   | {Euler_error_FUES: <14.6f} | {time_end_FUES[0]*1000: <24.6f} | {time_end_FUES[1]*1000: <23.6f} |\n"
        f"| DCEGM  | {Euler_error_DCEGM: <14.6f} | {time_end_DCEGM[0]*1000: <24.6f} | {time_end_DCEGM[1]*1000: <23.6f} |\n"
        f"| CONSAV | {Euler_error_CONSAV: <14.6f} | {time_end_CONSAV[0]*1000: <24.6f} | {time_end_CONSAV[1]*1000: <23.6f} |\n"
        "---------------------------------------------------------------------------------\n"
    )

    # 2. Plot and save value and policy on ref/unref. EGM grids
    plot_egrids(
        egrid_plot_age, e_grid_worker_unref, vf_work_unref, c_worker_unref,
        dela_unrefined, g_size_baseline, cp, tag='sigma0'
    )

    # 3. Plot consumption function for worker, before next period's work decision
    plot_cons_pol(c_refined_FUES, cp)

    # 4. Plot comparison with DC-EGM
    v_upper, v1_env, vf_interp_fues, a_interp_fues, m_upper, a1_env = \
        plot_dcegm_cf(
            egrid_plot_age, g_size_baseline, e_grid_worker_unref, vf_work_unref,
            c_worker_unref, dela_unrefined, cp.asset_grid_A, cp, tag='sigma0', plot=True
        )