import os, sys


import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl
from matplotlib.ticker import FormatStrFormatter
from quantecon.markov import MarkovChain
import dill as pickle
from itertools import groupby

# Import local modules
cwd = os.getcwd()
sys.path.append('..')
os.chdir(cwd)

# local modules for Fella model
from examples.fella import ConsumerProblem, Operator_Factory, iterate_euler
from examples.fella import iterate_euler, welfare_loss_log_utility
from FUES.math_funcs import mask_jumps


def compare_methods_grid(fella_settings, mc, z_series, grid_sizes_A, 
                         grid_sizes_H, max_iter=100, tol=1e-03, n=3):
    """
    Compare the performance of FUES, DCEGM, and RFC over different grid sizes.

    Parameters
    ----------
    fella_settings : dict
        Settings for the FUES algorithm.
    mc : MarkovChain
        Markov chain instance for the consumer problem.
    z_series : np.ndarray
        Simulated series for the z values.
    grid_sizes_A : list
        List of asset grid sizes to compare.
    grid_sizes_H : list
        List of housing grid sizes to compare.
    max_iter : int, optional
        Maximum number of iterations for the solver, default is 100.
    tol : float, optional
        Tolerance for convergence, default is 1e-03.
    n : int, optional
        Number of repetitions to average results over, default is 4.

    Returns
    -------
    results_summary : list
        A list containing performance metrics for each grid size and method.
    latex_table : str
        A LaTeX-formatted table of the results.
    """
    results_summary = []

    for grid_size_A in grid_sizes_A:
        for grid_size_H in grid_sizes_H:
            # Update the grid size in the model parameters
            print(grid_size_A)
            cp = ConsumerProblem(
                r=fella_settings['r'],
                r_H=fella_settings['r_H'],
                beta=fella_settings['beta'],
                delta=fella_settings['delta'],
                Pi=fella_settings['Pi'],
                z_vals=fella_settings['z_vals'],
                b=float(fella_settings['b']),
                grid_max_A=float(fella_settings['grid_max_A']),
                grid_max_H=float(fella_settings['grid_max_H']),
                grid_size=grid_size_A,
                grid_size_H=grid_size_H,
                gamma_1=fella_settings['gamma_1'],
                xi=fella_settings['xi'],
                kappa=fella_settings['kappa'],
                phi=fella_settings['phi'],
                theta=fella_settings['theta']
            )

            for method in ['FUES', 'DCEGM', 'RFC']:
                best_time = np.inf
                best_euler_error = np.inf
                total_runtime = 0

                for _ in range(n):  # Run each method n times
                    start_time = time.time()
                    results = iterate_euler(cp, method=method, 
                                            max_iter=max_iter, tol=tol, 
                                            verbose=False)
                    runtime = time.time() - start_time
                    total_runtime += runtime

                    if results['UE_time'] < best_time:
                        best_time = results['UE_time']
                        best_results = results

                    # Calculate Euler error for each iteration
                    _, _, _, euler_error_fella = Operator_Factory(cp)
                    E_error, _, _, _ = euler_error_fella(
                        z_series,
                        results['post_state']['H_prime'],
                        results['state']['c'],
                        results['state']['a'],
                        results['post_state']['c']
                    )

                    if E_error < best_euler_error:
                        best_euler_error = E_error

                # Record the average total time and best UE_time
                avg_total_runtime = total_runtime / n
                results_summary.append({
                    'Grid_Size_A': grid_size_A,
                    'Grid_Size_H': grid_size_H,
                    'Method': method,
                    'Best_UE_time': best_time,
                    'Avg_Total_Runtime': avg_total_runtime,
                    'Best_Euler_Error': best_euler_error,
                    'Total_iterations': best_results['Total_iterations']
                })

                print(f"Grid Size A: {grid_size_A}, Grid Size H: {grid_size_H}, "
                      f"Method: {method}, Best UE Time: {best_time}, "
                      f"Best Euler Error: {best_euler_error}")

    # Save results summary as a pickle file
    with open('../results/fella_timings.pkl', 'wb') as file:
        pickle.dump(results_summary, file)

    # Create a LaTeX table
    latex_table = create_latex_table(results_summary)
    file_path = '../results/fella_timings.tex'

    with open(file_path, 'w') as file:
        file.write(latex_table)

    return results_summary, latex_table

from itertools import groupby

from itertools import groupby

def create_latex_table(results_summary):
    """
    Generates a LaTeX table that includes timing (in milliseconds) and 
    Euler error comparisons for RFC, FUES, and DCEGM methods across 
    different grid sizes (A and H), grouped by A where A spans multiple rows.

    The grid sizes will be sorted first by Grid_Size_A and then by Grid_Size_H.

    Parameters
    ----------
    results_summary : list
        A list containing performance metrics for each grid size and method.

    Returns
    -------
    table : str
        A LaTeX-formatted table as a string.
    """
    table = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    table += (
        "\\begin{tabular}{ccccc|ccc}\n\\toprule\n"
        "\\multirow{4}{*}{\\textit{Grid Size A}} & "
        "\\multirow{3}{*}{\\textit{Grid Size H}} & "
        "\\multicolumn{3}{c}{\\textbf{Timing (milliseconds)}} & "
        "\\multicolumn{3}{c}{\\textbf{Euler error (Log10)}} \\\\\n"
        " & & \\textbf{RFC} & \\textbf{FUES} & \\textbf{DCEGM} & "
        "\\textbf{RFC} & \\textbf{FUES} & \\textbf{DCEGM} \\\\\n"
        "\\midrule\n"
    )

    # Sort the results by Grid_Size_A first, then by Grid_Size_H
    results_summary = sorted(results_summary, key=lambda x: (x['Grid_Size_A'], x['Grid_Size_H']))

    # Group by Grid_Size_A
    for grid_size_A, group in groupby(results_summary, key=lambda x: x['Grid_Size_A']):
        group = list(group)
        grid_size_H_count = len(set([result['Grid_Size_H'] for result in group]))  # Count unique Grid_Size_H
        #rint(grid_size_H_count)

        # For each Grid_Size_A, iterate over the grid_size_H values and methods
        first_row = True
        for grid_size_H in sorted(set([result['Grid_Size_H'] for result in group])):
            # Initialize timing and Euler error values
            time_rfc, time_fues, time_dcegm = "-", "-", "-"
            error_rfc, error_fues, error_dcegm = "-", "-", "-"
            print(grid_size_H)

            # Loop through the methods and update times and errors for the correct Grid_Size_H
            for result in group:
                if result['Grid_Size_H'] == grid_size_H:
                    method = result['Method']
                    if method == 'RFC':
                        time_rfc = f"{result['Best_UE_time'] * 1000:.3f}"
                        error_rfc = f"{result['Best_Euler_Error']:.3f}"
                    elif method == 'FUES':
                        time_fues = f"{result['Best_UE_time'] * 1000:.3f}"
                        error_fues = f"{result['Best_Euler_Error']:.3f}"
                    elif method == 'DCEGM':
                        time_dcegm = f"{result['Best_UE_time'] * 1000:.3f}"
                        error_dcegm = f"{result['Best_Euler_Error']:.3f}"

            if first_row:
                # First row for this Grid_Size_A, use \multirow for Grid_Size_A
                table += (
                    f"\\multirow{{{grid_size_H_count}}}{{*}}{{{grid_size_A}}} & "
                    f"{grid_size_H} & {time_rfc} & {time_fues} & {time_dcegm} & "
                    f"{error_rfc} & {error_fues} & {error_dcegm} \\\\\n"
                )
                first_row = False
            else:
                # Subsequent rows for this Grid_Size_A (no need for multirow)
                table += (
                    f" & {grid_size_H} & {time_rfc} & {time_fues} & {time_dcegm} & "
                    f"{error_rfc} & {error_fues} & {error_dcegm} \\\\\n"
                )

        table += "\\midrule\n"

    # Finish the table
    table += "\\bottomrule\n\\end{tabular}\n"
    table += (
        "\\caption{\\small Speed and accuracy of FUES, DCEGM, and RFC "
        "across different grid sizes A and H.}\n\\end{table}"
    )

    return table




# Function to summarize results for a given housing grid size
def summarize_results(results_FUES, results_DCEGM, grid_size_H, z_series, cp, method_name="FUES"):
    """
    Summarizes and prints the performance comparison between FUES and DCEGM 
    for a given housing grid size.

    Parameters:
    -----------
    results_FUES : dict
        The result dictionary from the FUES iteration.
    results_DCEGM : dict
        The result dictionary from the DCEGM iteration.
    grid_size_H : int
        The housing grid size for the comparison.
    z_series : array
        The Markov chain z-series used for the simulation.
    cp : ConsumerProblem
        The consumer problem instance.
    method_name : str, optional
        The method used (FUES/DCEGM), default is 'FUES'.
    """
    
    euler_error_fella = Operator_Factory(cp)[3]
    
    # Calculate Euler error for FUES
    euler_error_FUES, _, v_val_est_fues, cons_fues = euler_error_fella(
        z_series, results_FUES['post_state']['H_prime'],
        results_FUES['state']['c'], results_FUES['state']['a'], 
        results_FUES['post_state']['c']
    )
    
    # Calculate Euler error for DCEGM
    euler_error_DCEGM, _, v_val_est_dcegm, cons_dcegm = euler_error_fella(
        z_series, results_DCEGM['post_state']['H_prime'],
        results_DCEGM['state']['c'], results_DCEGM['state']['a'], 
        results_DCEGM['post_state']['c']
    )
    
    # Calculate welfare loss
    CV = welfare_loss_log_utility(v_val_est_fues, v_val_est_dcegm, cons_dcegm)
    
    # Print a formatted summary of the results
    print(f"----- Comparison Summary for Housing Grid Size H = {grid_size_H} -----")
    print(f"FUES:")
    print(f"  - UE Time: {results_FUES['UE_time']} seconds")
    print(f"  - Total iterations: {results_FUES['Total_iterations']}")
    print(f"  - Euler Error: {euler_error_FUES}")
    print(f"  - Mean Value Function Estimate: {np.nanmean(v_val_est_fues)}")
    
    print(f"DCEGM:")
    print(f"  - UE Time: {results_DCEGM['UE_time']} seconds")
    print(f" - Total iterations: {results_DCEGM['Total_iterations']}")
    print(f"  - Euler Error: {euler_error_DCEGM}")
    print(f"  - Mean Value Function Estimate: {np.nanmean(v_val_est_dcegm)}")
    
    print(f"Consumption Equivalent Welfare Loss: {CV}")
    print("-------------------------------------------------------------\n")



# Main block of code
if __name__ == "__main__":
    
    import yaml

    # 0. Load the settings from the YAML file/ set up shocks for Euler error
    # Load the YAML file
    with open('../settings/settings.yml', 'r') as file:
        settings = yaml.safe_load(file)

    # Extract the 'fella' section from the settings
    fella_settings = settings['fella']

    mc = MarkovChain(fella_settings['Pi'])
    z_series = mc.simulate(ts_length=1000000, init=1)

    # 1. Produce timings table from saved results. 
    # Uncomment the following lines if you need to load a pickle file 
    # with timing results or generate a LaTeX table from it.

    with open('../results/fella_timings.pkl', 'rb') as file:
         results_summary = pickle.load(file)
    
    latex_table = create_latex_table(results_summary)
    print(latex_table)
    file_path = '../results/fella_timings.tex'
    with open(file_path, 'w') as file:
        file.write(latex_table)

    # 2. Run the FUES, DCEGM, and RFC methods for different grid sizes
    # Grid sizes for comparison

    grid_sizes_A = [500, 1000, 2000,3000]
    grid_sizes_H = [5, 10,15]

    # Uncomment this block if you want to compare methods across grids.
    #compare_methods_grid(fella_settings, mc, z_series, grid_sizes_A, 
    #                     grid_sizes_H, max_iter=200, tol=1e-03)

    # 3. Run the FUES and DCEGM methods for a specific grid size and plot
    # Instantiate the consumer problem with parameters
    cp_6 = ConsumerProblem(
        r=0.0346,
        r_H=0,
        beta=0.93,
        delta=0,
        Pi=((0.99, 0.01, 0), (0.01, 0.98, 0.01), (0, 0.09, 0.91)),
        z_vals=(0.2, 0.526, 4.66),
        b=1e-200,
        grid_max_A=30,
        grid_max_H=5,
        grid_size=2000,
        grid_size_H=6,
        gamma_1=0,
        xi=0,
        kappa=0.077,
        phi=0.09,
        theta=0.77,
        m_bar = 1.5, 
        lb = 4
    )

    cp_10 = ConsumerProblem(
        r=0.0346,
        r_H=0,
        beta=0.93,
        delta=0,
        Pi=((0.99, 0.01, 0), (0.01, 0.98, 0.01), (0, 0.09, 0.91)),
        z_vals=(0.2, 0.526, 4.66),
        b=1e-200,
        grid_max_A=30,
        grid_max_H=5,
        grid_size=2000,
        grid_size_H=12,
        gamma_1=0,
        xi=0,
        kappa=0.077,
        phi=0.09,
        theta=0.77, 
        m_bar = 1.5, 
        lb = 4
    )


    # 4. Run the VFI (TBA)
    bellman_operator, euler_operator, condition_V, euler_error_fella = \
        Operator_Factory(cp_6)

    shape = (len(cp_6.z_vals), len(cp_6.asset_grid_A), len(cp_6.asset_grid_H))
    V_init = np.ones(shape)
    a_policy = np.empty(shape)
    h_policy = np.empty(shape)
    value_func = np.empty(shape)

    bell_error = 0
    bell_toll = 1e-3
    iteration = 0
    new_V = V_init
    max_iter = 200

    sns.set(style="whitegrid", rc={"font.size": 10, "axes.titlesize": 10, 
                                   "axes.labelsize": 10})

    # Solve via VFI and plot
    start_time = time.time()
    while bell_error > bell_toll and iteration < max_iter:
        V = np.copy(new_V)
        a_new_policy, h_new_policy, V_new_policy, _, _, _, new_c_policy = \
            bellman_operator(iteration, V)

        new_V, _, _ = condition_V(V_new_policy, V_new_policy, V_new_policy)
        a_policy, h_policy, value_func = np.copy(a_new_policy), \
                                         np.copy(h_new_policy), \
                                         np.copy(new_V)

        bell_error = np.max(np.abs(V - value_func))
        print(f"Iteration {iteration + 1}, error is {bell_error}")
        iteration += 1

    print(f"VFI in {time.time() - start_time} seconds")

    ####################################################################
    # Run FUES and DCEGM for grid size H = 6
    results_FUES_6 = iterate_euler(cp_6, method='FUES', max_iter=200, tol=1e-03, verbose=True)
    results_DCEGM_6 = iterate_euler(cp_6, method='DCEGM', max_iter=200, tol=1e-03, verbose=True)
    

    ####################################################################
    # Run FUES and DCEGM for grid size H = 10
    results_FUES_10 = iterate_euler(cp_10, method='FUES', max_iter=200, tol=1e-03, verbose=True)
    results_DCEGM_10 = iterate_euler(cp_10, method='DCEGM', max_iter=200, tol=1e-03, verbose=True)
    

    # 5. Summarize the results and plot the policy functions
    # Summarize the results for H = 6
    summarize_results(results_FUES_6, results_DCEGM_6, 6, z_series, cp_6)
    
    # Summarize the results for H = 10
    summarize_results(results_FUES_10, results_DCEGM_10, 10, z_series, cp_10)
    
    # Set seaborn style for the plot
    sns.set(style="whitegrid", rc={"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10})

    # Plotting final policy functions
    fig, ax = pl.subplots(1, 2, figsize=(12, 6))

    # Axis labels, appearance, and titles for both subplots
    for a in ax:
        a.set_xlabel('Assets (t)', fontsize=11)
        a.set_ylabel('Consumption', fontsize=11)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.tick_params(axis='x', labelsize=9)
        a.tick_params(axis='y', labelsize=9)
        a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        a.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Colors for FUES, avoiding black
    fues_colors = ['blue', 'red', 'green']  # Picked distinct colors

    # Plotting for grid size H = 6 (Left subplot)
    for i, col, lab in zip([1, 2, 3], fues_colors, ['H = low', 'H = med.', 'H = high']):
        # Mask FUES and DCEGM data for jumps
        fues_masked_6 = mask_jumps(results_FUES_6['state']['c'][1, :, i])
        dcegm_masked_6 = mask_jumps(results_DCEGM_6['state']['c'][1, :, i])

        # FUES: Solid colored lines
        ax[0].plot(cp_6.asset_grid_M, fues_masked_6,
                color=col, linestyle='-', label=f'FUES - {lab}')
        # DCEGM: Dotted black lines
        ax[0].plot(cp_6.asset_grid_M, dcegm_masked_6,
                color='black', linestyle=':', label=f'DCEGM - {lab}')

    ax[0].set_title("Grid Size H = 6", fontsize=11)
    ax[0].set_xlim([0, 15])
    ax[0].set_ylim([0, 2])

    # Plotting for grid size H = 10 (Right subplot)
    for i, col, lab in zip([1, 2, 3], fues_colors, ['H = low', 'H = med.', 'H = high']):
        # Mask FUES and DCEGM data for jumps
        fues_masked_10 = mask_jumps(results_FUES_10['state']['c'][1, :, int(i * 2)])
        dcegm_masked_10 = mask_jumps(results_DCEGM_10['state']['c'][1, :, int(i * 2)])

        # FUES: Solid colored lines
        ax[1].plot(cp_6.asset_grid_M, fues_masked_10,
                color=col, linestyle='-', label=f'FUES - {lab}')
        # DCEGM: Dotted black lines
        ax[1].plot(cp_6.asset_grid_M, dcegm_masked_10,
                color='black', linestyle=':', label=f'DCEGM - {lab}')

    ax[1].set_title("Grid Size H = 10", fontsize=11)
    ax[1].set_xlim([0, 15])
    ax[1].set_ylim([0, 2])

    # Combined legend placed at the bottom with padding
    # Combined legend placed at the bottom with padding below the plot
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles[:6], labels[:6], loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.1))

    # Adjust layout to give space for the legend with padding below the plot
    fig.tight_layout(rect=[0, 0, 1, 0.9])  # Padding to make room for the legend

    pl.savefig('../results/plots/fella/Fella_policy.png')


    # 7. Plot raw and refined EGM grids for both methods
    fig, ax = pl.subplots(1, 2, figsize=(12, 6))
    sns.set(style="white", rc={"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

    ax[1].set_xlabel('Assets (t)', fontsize=11)
    ax[1].set_ylabel('Value function', fontsize=11)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].tick_params(axis='x', labelsize=9)
    ax[1].tick_params(axis='y', labelsize=9)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # Plot unrefined and refined value functions for DCEGM and FUES
    for i, col, lab in zip([1, 4], ['blue', 'red'], ['H = low', 'H = med.', 'H = high']):
        ax[0].scatter(results_DCEGM_6['EGM']['unrefined']['e'][1, :, i], results_DCEGM_6['EGM']['unrefined']['v'][1, :, i],
                      color=col, label=f'Unrefined {lab}', linewidth=0.75, s=20, edgecolors='r', facecolors='none')
        ax[0].scatter(results_DCEGM_6['EGM']['refined']['e'][f'1-{i}'], results_DCEGM_6['EGM']['refined']['v'][f'1-{i}'],
                      label=f'Refined {lab}', marker='x', linewidth=0.75, s=15, color='blue')

        ax[1].scatter(results_FUES_6['EGM']['unrefined']['e'][1, :, i], results_FUES_6['EGM']['unrefined']['v'][1, :, i],
                      color=col, label=f'Unrefined {lab}', linewidth=0.75, s=20, edgecolors='r', facecolors='none')
        ax[1].scatter(results_FUES_6['EGM']['refined']['e'][f'1-{i}'], results_FUES_6['EGM']['refined']['v'][f'1-{i}'],
                      label=f'Refined {lab}', marker='x', linewidth=0.75, s=15, color='blue')

    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].set_title("DCEGM: Unrefined vs Refined", fontsize=11)

    ax[1].legend(frameon=False, prop={'size': 10})
    ax[1].set_title("FUES: Unrefined vs Refined", fontsize=11)

    fig.tight_layout()
    pl.savefig('../results/plots/fella/Fella_policy_refined_vs_unrefined.png')
    pl.show()