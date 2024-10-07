"""
Author: Akshay Shanker, University of New South Wales, 
a.shanker@unsw.edu.au

Script to plot NEGM and RFC solutions for Application 3 in Dobrescu and 
Shanker (2024) (FUES). Model with a durable and non-durable continuous asset. 
"""

import numpy as np
import time
import yaml
from interpolation.splines import UCGrid, eval_linear
from interpolation.splines import extrap_options as xto
from mpi4py import MPI
import os
import sys

# Import local modules
cwd = os.getcwd()
sys.path.append('..')
os.chdir(cwd)
from examples.durables.durables import ConsumerProblem, Operator_Factory
from examples.durables.plot import plot_pols, plot_grids
from FUES.math_funcs import f, interp_as

def euler_housing(results, cp):
    """
    Compute Euler errors for the housing model.

    Parameters
    ----------
    results : dict
        Results from the solver, containing policy functions.
    cp : ConsumerProblem
        The consumer problem instance.

    Returns
    -------
    euler : np.array
        Euler errors across the state space.
    """
    ug_grid_all = UCGrid((cp.b, cp.grid_max_A, cp.grid_size),
                         (cp.b, cp.grid_max_H, len(cp.asset_grid_H)))

    euler = np.zeros((cp.T, len(cp.z_vals), len(cp.asset_grid_A),
                      len(cp.asset_grid_H)))
    euler.fill(np.nan)
    cpdiff = np.zeros(euler.shape)
    cpdiff.fill(np.nan)
    a_grid = cp.asset_grid_A
    h_grid = cp.asset_grid_H
    asset_grid_WE = cp.asset_grid_WE
    y_func = cp.y_func
    R = cp.R
    R_H = cp.R_H
    delta = cp.delta

    if cp.stat:
        t0 = cp.T - 1 - 50
        T = cp.T - 1
    else:
        t0 = cp.t0
        T = cp.T

    # Loop over grid
    for t in range(t0, T):
        for i_h in range(len(h_grid)):
            for i_a in range(len(a_grid)):
                for i_z in range(len(cp.z_vals)):
                    h = h_grid[i_h]
                    a = a_grid[i_a]
                    z = cp.z_vals[i_z]
                    D_adj = results[t]['D'][i_z, i_a, i_h]

                    if D_adj == 1:
                        wealth = R * a + R_H * h * (1 - delta) + y_func(t, z)
                        a_prime = interp_as(asset_grid_WE,
                                            results[t]['Aadj'][i_z, :],
                                            np.array([wealth]))[0]
                        hnxt = interp_as(asset_grid_WE,
                                         results[t]['Hadj'][i_z, :],
                                         np.array([wealth]))[0]
                        c = interp_as(asset_grid_WE,
                                      results[t]['Cadj'][i_z, :],
                                      np.array([wealth]))[0]
                    if D_adj == 0:
                        wealth = R * a + y_func(t, z)
                        hnxt = h
                        a_prime = interp_as(a_grid,
                                            results[t]['Akeeper'][i_z, :, i_h],
                                            np.array([wealth]))[0]
                        c = interp_as(a_grid,
                                      results[t]['Ckeeper'][i_z, :, i_h],
                                      np.array([wealth]))[0]

                    if a_prime <= 0.1 or c <= 0.1:
                        continue

                    rhs = 0
                    for i_eta in range(len(cp.z_vals)):
                        c_plus = eval_linear(ug_grid_all,
                                             results[t + 1]['C'][i_z],
                                             np.array([a_prime, hnxt]),
                                             xto.LINEAR)
                        rhs += cp.Pi[i_z, i_eta] * cp.beta * cp.R * \
                               cp.du_c(c_plus)

                    lambda_h_plus = eval_linear(
                        ug_grid_all, results[t + 1]['ELambdaHnxt'][i_z],
                        np.array([a_prime, hnxt])
                    )
                    lhs = (cp.du_h(hnxt) + lambda_h_plus) / (1 + cp.tau)
                    euler_raw2 = c - cp.du_c_inv(lhs)
                    euler[t, i_z, i_a, i_h] = np.log10(np.abs(
                        euler_raw2 / c) + 1e-16)
    return euler

def timing(solver, cp, rep=4, do_print=False, method='DCEGM'):
	"""
	Run the solver and track the best time, average Euler error, 
	and number of iterations to convergence.
	
	Parameters
	----------
	solver : callable
		The solver function to use (e.g., solveNEGM, solveEGM).
	cp : ConsumerProblem
		The consumer problem instance.
	rep : int, optional
		Number of repetitions to average results over, default is 4.
	do_print : bool, optional
		Whether to print intermediate results, default is False.
	method : str, optional
		Method to compare, default is 'DCEGM'.
	
	Returns
	-------
	dict
		The solution with the best time, Euler error, and iteration count.
	"""
	ug_grid_all = UCGrid((cp.b, cp.grid_max_A, cp.grid_size),
						 (cp.b, cp.grid_max_H, len(cp.asset_grid_H)))

	time_best = np.inf
	iter_best = None

	for i in range(rep):
		solution = solver(cp, method=method)
		euler = euler_housing(solution, cp)
		tot_time = solution['avg_time']
		iterations = solution.get('iterations', None)  # Get iteration count

		if do_print:
			print(f'{i}: {tot_time:.2f} secs, euler: {np.nanmean(euler):.3f}')
			print(f'Iterations: {iterations}')

		if tot_time < time_best:
			time_best = tot_time
			model_best = solution
			iter_best = iterations

	model_best['euler'] = euler
	model_best['iterations'] = iter_best

	return model_best
		

def initVal(cp):
	shape = (len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
	EVnxt = np.empty(shape)
	ELambdaAnxt = np.empty(shape)
	ELambdaHnxt = np.empty(shape)

	for state in range(len(cp.X_all)):
		a = cp.asset_grid_A[cp.X_all[state][1]]
		h = cp.asset_grid_H[cp.X_all[state][2]]
		i_a = cp.X_all[state][1]
		i_h = cp.X_all[state][2]
		i_z = cp.X_all[state][0]
		z = cp.z_vals[i_z]

		EVnxt[i_z, i_a, i_h] = cp.term_u(
			cp.R_H * (1 - cp.delta) * h + cp.R * a)
		ELambdaAnxt[i_z, i_a, i_h] = cp.beta * cp.R * cp.term_du(
			cp.R_H * (1 - cp.delta) * h + cp.R * a)
		ELambdaHnxt[i_z, i_a, i_h] = cp.beta * cp.beta * cp.R_H * cp.term_du(
			cp.R_H * (1 - cp.delta) * h + cp.R * a) * (1 - cp.delta)

	return EVnxt, ELambdaAnxt, ELambdaHnxt


def solveVFI(cp, verbose=False):
	iterVFI, _, condition_V, _ = Operator_Factory(cp)

	# Initial VF
	EVnxt, _, _ = initVal(cp)

	# Start Bellman iteration
	t = cp.T
	results = {}
	times = []

	while t >= cp.t0:
		results[t] = {}

		start = time.time()
		bellman_sol = iterVFI(t, EVnxt)

		Vcurr, Anxt, Hnxt, Cnxt, Aadj, Hadj = bellman_sol

		# Store results
		results[t]["VF"] = Vcurr
		results[t]["C"] = Cnxt
		results[t]["A"] = Anxt
		results[t]["H"] = Hnxt
		results[t]["Hadj"] = Hadj
		results[t]["Aadj"] = Aadj

		# Condition the value function
		EVnxt, _, _ = condition_V(Vcurr, Vcurr, Vcurr)

		if verbose:
			print(f"Bellman iteration no. {t}, time is {time.time() - start}")
			if t< cp.T: print(f"Error at age {t} is {error}")

		if t < cp.T:
			times.append(time.time() - start)

			error = np.max(np.abs(Vcurr - results[t + 1]["VF"]))

		t = t - 1

	results['avg_time'] = np.mean(times)

	return results

def solveEGM(cp, LS=True, verbose=True, method = 'FUES'):
	_, iterEGM, condition_V, _ = Operator_Factory(cp)

	# Initial values
	EVnxt, ELambdaAnxt, ELambdaHnxt = initVal(cp)

	t = cp.T
	times = []
	results = {}

	error = 1

	while t >= cp.t0 and error > cp.tol_bel:
		start = time.time()

		(Vcurr, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, AdjPol,
		 KeeperPol, EGMGrids) = iterEGM(t, EVnxt, ELambdaAnxt,\
								   		ELambdaHnxt, method = method, m_bar = cp.m_bar,
		 )
					
		EVnxt, ELambdaAnxt, ELambdaHnxt = condition_V(
											Vcurr, LambdaAcurr, LambdaHcurr
		)
		
		if t < cp.T:
			times.append(time.time() - start)

		results[t] = {}
		results[t]["D"] = Dnxt
		results[t]["C"] = Cnxt
		results[t]["H"] = Hnxt
		results[t]["VF"] = Vcurr
		results[t]["Hadj"] = AdjPol['H']
		results[t]["Aadj"] = AdjPol['A']
		results[t]["Cadj"] = AdjPol['C']
		results[t]["Vadj"] = KeeperPol['V']
		results[t]["Ckeeper"] = KeeperPol['C']
		results[t]["Akeeper"] = KeeperPol['A']
		results[t]["EGMGrids"] = EGMGrids
		results[t]["ELambdaHnxt"] = ELambdaHnxt

		if t < cp.T:
			error = np.max(np.abs(Vcurr - results[t + 1]["VF"]))

		if verbose:
			print(f"EGM age {t}, time is {time.time() - start}")
			if t< cp.T: print(f"Error at age {t} is {error}")

		t = t - 1

	results['avg_time'] = np.mean(times)
	results['iterations'] = cp.T1 -t

	return results

def solveNEGM(cp, LS=True, verbose=True, method = 'DCEGM'):
	iterVFI, iterEGM, condition_V, iterNEGM = Operator_Factory(cp)

	# Initial values
	EVnxt, ELambdaAnxt, ELambdaHnxt = initVal(cp)

	# Results dictionaries
	results = {}
	times = []
	t = cp.T
	error =1
	while t >= cp.t0 and error > cp.tol_bel:
		results[t] = {}
		start = time.time()

		(Vcurr, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, AdjPol, KeeperPol) = iterNEGM(
			EVnxt, ELambdaAnxt, ELambdaHnxt, t)

		

		EVnxt, ELambdaAnxt, ELambdaHnxt = condition_V(
			Vcurr, LambdaAcurr, LambdaHcurr)

		if t < cp.T:
			times.append(time.time() - start)

		results[t] = {}
		results[t]["D"] = Dnxt
		results[t]["C"] = Cnxt
		results[t]["H"] = Hnxt
		results[t]["VF"] = Vcurr
		results[t]["Hadj"] = AdjPol['H']
		results[t]["Aadj"] = AdjPol['A']
		results[t]["Cadj"] = AdjPol['C']
		results[t]["Vadj"] = KeeperPol['V']
		results[t]["Ckeeper"] = KeeperPol['C']
		results[t]["Akeeper"] = KeeperPol['A']
		results[t]["ELambdaHnxt"] = ELambdaHnxt

		if t < cp.T:
			error = np.max(np.abs(Vcurr - results[t + 1]["VF"]))

		if verbose:
			print(f"NEGM age {t}, time {time.time() - start}")
			if t< cp.T: print(f"Error at age {t} is {error}")
			
		t = t - 1

	results['avg_time'] = np.mean(times)
	results['iterations'] = cp.T1 -t

	return results

def compare_grids_and_tau(cp_settings, tau_values, grid_sizes, max_iter=200, tol=1e-03, rep=4, methods=['DCEGM', 'FUES', 'RFC']):
	"""
	Compare the performance of FUES, NEGM, and RFC over different grid sizes and tau values using MPI.

	Parameters
	----------
	cp_settings : dict
		Settings for the consumer problem.
	tau_values : list
		List of `tau` values to compare.
	grid_sizes : list
		List of grid sizes (same for assets and housing) to compare.
	max_iter : int, optional
		Maximum number of iterations for the solver, default is 200.
	tol : float, optional
		Tolerance for convergence, default is 1e-03.
	rep : int, optional
		Number of repetitions to average results over, default is 4.
	methods : list
		Methods to compare, default is ['DCEGM', 'FUES', 'RFC'].

	Returns
	-------
	results_summary : list
		A list containing performance metrics for each grid size, tau, and method.
	"""
	# Initialize MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()  # Get the rank (ID) of the current process
	size = comm.Get_size()  # Get the total number of processes

	# Distribute work based on rank
	total_combinations = len(tau_values) * len(grid_sizes) * len(methods)
	combinations_per_process = total_combinations // size  # Split work equally
	leftover_combinations = total_combinations % size  # Handle leftover combinations

	# Each process works on a subset of combinations
	results_summary = []

	# Flatten the combinations of tau, grid_size, and method for easy distribution
	combinations = [(tau, grid_size, method) for tau in tau_values for grid_size in grid_sizes for method in methods]
	
	# Get the range of combinations this process will handle
	start_index = rank * combinations_per_process
	end_index = start_index + combinations_per_process
	if rank < leftover_combinations:
		start_index += rank
		end_index += rank + 1
	else:
		start_index += leftover_combinations
		end_index += leftover_combinations

	# Each process works on its subset of combinations
	for tau, grid_size, method in combinations[start_index:end_index]:
		# Update the tau and grid size in the model parameters
		cp = ConsumerProblem(cp_settings,
							 r=cp_settings['r'],
							 sigma=cp_settings['sigma'],
							 r_H=cp_settings['r_H'],
							 beta=cp_settings['beta'],
							 alpha=cp_settings['alpha'],
							 delta=cp_settings['delta'],
							 Pi=cp_settings['Pi'],
							 z_vals=cp_settings['z_vals'],
							 b=np.float64(cp_settings['b']),
							 grid_max_A=cp_settings['grid_max_A'],
							 grid_max_WE=cp_settings['grid_max_WE'],
							 grid_size_W=cp_settings['grid_size_W'],
							 grid_max_H=cp_settings['grid_max_H'],
							 grid_size=grid_size,
							 grid_size_H=grid_size,
							 gamma_c=cp_settings['gamma_c'],
							 chi=cp_settings['chi'],
							 tau=tau,
							 K=cp_settings['K'],
							 tol_bel=np.float64(cp_settings['tol_bel']),
							 m_bar=cp_settings['M_bar'],
							 T=cp_settings['max_iter'],
							 theta=cp_settings['theta'], 
							 t0=0, 
							 root_eps= np.float64(cp_settings['root_eps']), 
							 stat=True
							 )

		# Timing and Euler error calculations
		best_time = np.inf
		best_euler_error = np.inf
		best_iterations = None
		total_runtime = 0

		for _ in range(rep):  # Run each method `rep` times
			if method == 'DCEGM':
				solution = solveNEGM(cp, method=method)
			else:
				solution = solveEGM(cp, method=method)

			euler = euler_housing(solution, cp)
			runtime = solution['avg_time']
			iterations = solution['iterations']
			total_runtime += runtime

			if runtime < best_time:
				best_time = runtime
				best_euler = euler
				best_iterations = iterations

		avg_time = total_runtime / rep
		avg_euler_error = np.nanmean(best_euler)

		results_summary.append({
			'Tau': tau,
			'Grid_Size': grid_size,
			'Method': method,
			'Avg_Time': avg_time,
			'Euler_Error': avg_euler_error,
			'Iterations': best_iterations
		})

	# Gather results from all processes
	all_results = comm.gather(results_summary, root=0)

	# If root process, combine and process results
	if rank == 0:
		combined_results = [item for sublist in all_results for item in sublist]
		latex_table = create_latex_table(combined_results)
		return combined_results, latex_table
	else:
		return None, None

def create_latex_table(results_summary):
	"""
	Generate a LaTeX table comparing timing, Euler errors, and number of iterations 
	for different grid sizes and tau values across the methods.
	
	Parameters
	----------
	results_summary : list
		A list containing performance metrics for each grid size, tau, and method.
	
	Returns
	-------
	latex_table : str
		A LaTeX-formatted table.
	"""
	table = "\\begin{table}[htbp]\n\\centering\n\\small\n"
	table += (
		"\\begin{tabular}{cccccc|cc}\n\\toprule\n"
		"\\multirow{2}{*}{\\textit{Tau}} & "
		"\\multirow{2}{*}{\\textit{Grid Size A}} & "
		"\\multirow{2}{*}{\\textit{Grid Size H}} & "
		"\\multirow{2}{*}{\\textit{Method}} & "
		"\\textbf{Avg. Time (sec)} & \\textbf{Euler Error} & \\textbf{Iterations} \\\\\n"
		"\\midrule\n"
	)

	for result in results_summary:
		table += (
			f"{result['Tau']} & {result['Grid_Size_A']} & {result['Grid_Size_H']} & "
			f"{result['Method']} & {result['Avg_Time']:.2f} & {result['Euler_Error']:.3f} & "
			f"{result['Iterations']} \\\\\n"
		)

	table += "\\bottomrule\n\\end{tabular}\n"
	table += "\\caption{\\small Comparison of Timing, Euler Errors, and Iterations for FUES, DCEGM, and RFC}\n\\end{table}"
	
	return table

if __name__ == "__main__":
	
	# Read settings
	with open("../settings/settings.yml", "r") as stream:
		settings = yaml.safe_load(stream) 

	# 1. Error and timing comparisons across grid sizes and tau values
	# Tau values and grid sizes to compare
	cp_settings = settings['durables']
	
	tau_values = [0.03, 0.07, 0.15]
	grid_sizes_A = [300, 500, 1000,2000]

	# Run comparison across grid sizes and tau values
	results, latex_table = compare_grids_and_tau(cp_settings, tau_values, grid_sizes_A)

	if MPI.COMM_WORLD.Get_rank() == 0:
		print(latex_table)
		with open("../results/comparison_table.tex", "w") as f:
			f.write(latex_table)

	# 2. Solve the model in single process interface for plotting and single table 

	if MPI.COMM_WORLD.Get_rank() == 0:

		

		cp = ConsumerProblem(cp_settings,
							r=cp_settings['r'],
							sigma=cp_settings['sigma'],
							r_H=cp_settings['r_H'],
							beta=cp_settings['beta'],
							alpha=cp_settings['alpha'],
							delta=cp_settings['delta'],
							Pi= cp_settings['Pi'],
							z_vals= cp_settings['z_vals'],
							b= 1e-200,
							grid_max_A= cp_settings['grid_max_A'],
							grid_max_WE= cp_settings['grid_max_WE'],
							grid_size_W= 300,
							grid_max_H= cp_settings['grid_max_H'],
							grid_size= 300,
							grid_size_H= 300,
							gamma_c= cp_settings['gamma_c'],
							chi= cp_settings['chi'],
							tau=cp_settings['tau'],
							K=cp_settings['K'],
							tol_bel = np.float64(cp_settings['tol_bel']),
							m_bar=	1.51,
							T = 200,
							theta= cp_settings['theta'], t0 =0, root_eps=1e-2, stat= True
		)
		
		#print(cp.stat)
		
		# 0. Solve with Bellman 
		#iterVFI, iterEGM, condition_V, NEGM = Operator_Factory(cp)
		#pickle.dump(bell_results, open("bell_results_300.p", "wb"))
		#bell_results = pickle.load(open("bell_results_300.p", "rb"))

		# 1. Solve using NEGM and EGM 
		
		NEGMRes = timing(solveNEGM, cp, rep =1, method = 'DCEGM')
		EGMRes_fues = timing(solveEGM, cp, rep =1, method = 'FUES')
		EGMRes_rfc = timing(solveEGM, cp, rep =1, method = 'RFC')

		NEGMRes['label'] = 'NEGM'
		EGMRes_fues['label'] = 'EGM_FUES'
		EGMRes_rfc['label'] = 'EGM_RFC'
	
		# 2a. Basic policy function plots
		plot_pols(cp,EGMRes_fues, NEGMRes, 58, [100,150,200])

		# 2.b. plot of adjuster endog. grids 
		plot_grids(EGMRes_fues,cp,term_t = 57)
			
		# 3. Euler errors
		eulerNEGM = euler_housing(NEGMRes, cp)
		eulerEGM = euler_housing(EGMRes_fues, cp)
		eulerRFCEGM = euler_housing(EGMRes_rfc, cp)
		print("NEGM Euler error is {}".format(np.nanmean(eulerNEGM)))
		print("EGM Euler error is {}".format(np.nanmean(eulerEGM)))
		print("RFCEGM Euler error is {}".format(np.nanmean(eulerRFCEGM)))

		# 4. Tabulate timing and errors with latex table 
		print("NEGM Time is {}".format(NEGMRes['avg_time']))
		print("EGM Time is {}".format(EGMRes_fues['avg_time']))
		print("RFCEGM Time is {}".format(EGMRes_rfc['avg_time']))

		lines = []
		txt = '| All (average)'
		txt += f' | {np.nanmean(eulerEGM):.3f}'
		txt += f' | {np.nanmean(eulerNEGM):.3f}'
		txt += ' |\n'
		lines.append(txt)

		txt = '| 5th percentile'
		txt += f' | {np.nanpercentile(eulerEGM,5):.3f}'
		txt += f' | {np.nanpercentile(eulerNEGM,5):.3f}'
		txt += ' |\n'
		lines.append(txt)

		txt = '| 95th percentile'
		txt += f' | {np.nanpercentile(eulerEGM,95):.3f}'
		txt += f' | {np.nanpercentile(eulerNEGM,95):.3f}'
		txt += ' |\n'
		lines.append(txt)

		txt = '| Median'
		txt += f' | {np.nanpercentile(eulerEGM,50):.3f}'
		txt += f' | {np.nanpercentile(eulerNEGM,50):.3f}'
		txt += ' |\n'
		lines.append(txt)

		txt = '| Avg. time per iteration (sec.)'
		txt += f' | {EGMRes_fues["avg_time"]:.2f}'
		txt += f' | {NEGMRes["avg_time"]:.2f}'
		txt += ' |\n'

		lines.append(txt)
		lines.insert(0, '| Method | EGM | NEGM |\n')

		with open("../results/durables_single_result.tex", "w") as f:
			f.writelines(lines)