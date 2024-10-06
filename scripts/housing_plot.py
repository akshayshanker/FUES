"""
Author: Akshay Shanker, University of New South Wales, a.shanker@unsw.edu.au

Script to plot NEGM and RFC solutions for Application 2 in Dobrescu and Shanker (2024)
Model with continuous housing and frictions
"""

import numpy as np
import time
import yaml
from interpolation.splines import UCGrid, eval_linear
from interpolation.splines import extrap_options as xto


import os, sys

# Import local modules
cwd = os.getcwd()
sys.path.append('..')
os.chdir(cwd)
# local modules
from examples.durables.durables import ConsumerProblem, Operator_Factory
from examples.durables.plot import plot_pols, plot_grids
from FUES.math_funcs import f, interp_as


#@njit
def euler_housing(results, cp):

    ug_grid_all = UCGrid((cp.b, cp.grid_max_A, cp.grid_size),
                         (cp.b, cp.grid_max_H, len(cp.asset_grid_H)))

    euler = np.zeros((cp.T, len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H)))
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

    # loop over grid
    for t in range(cp.t0, cp.T - 1):
        for i_h in range(len(h_grid)):
            for i_a in range(len(a_grid)):
                for i_z in range(len(cp.z_vals)):

                    h = h_grid[i_h]
                    a = a_grid[i_a]
                    z = cp.z_vals[i_z]

                    D_adj = results[t]['D'][i_z, i_a, i_h]
                    if D_adj == 1:
                        wealth = R * a + R_H * h * (1 - delta) + y_func(t, z)

                        a_prime = interp_as(asset_grid_WE, results[t]['Aadj'][i_z, :],
                                            np.array([wealth]))[0]
                        hnxt = interp_as(asset_grid_WE, results[t]['Hadj'][i_z, :],
                                         np.array([wealth]))[0]
                        c = interp_as(asset_grid_WE, results[t]['Cadj'][i_z, :],
                                      np.array([wealth]))[0]

                    if D_adj == 0:
                        wealth = R * a + y_func(t, z)
                        hnxt = h
                        a_prime = interp_as(a_grid, results[t]['Akeeper'][i_z, :, i_h],
                                            np.array([wealth]))[0]
                        c = interp_as(a_grid, results[t]['Ckeeper'][i_z, :, i_h],
                                      np.array([wealth]))[0]

                    if a_prime <= 0.1:
                        continue
                    if c <= 0.1:
                        continue

                    rhs = 0
                    for i_eta in range(len(cp.z_vals)):

                        c_plus = eval_linear(ug_grid_all, results[t + 1]['C'][i_z],
                                             np.array([a_prime, hnxt]), xto.LINEAR)

                        # accumulate
                        rhs += cp.Pi[i_z, i_eta] * cp.beta * cp.R * cp.du_c(c_plus)

                    # euler error
                    lambda_h_plus = eval_linear(ug_grid_all, results[t + 1]['ELambdaHnxt'][i_z],
                                                np.array([a_prime, hnxt]))
                    lhs = (cp.du_h(hnxt) + lambda_h_plus) / (1 + cp.tau)

                    euler_raw2 = c - cp.du_c_inv(lhs)
                    euler[t, i_z, i_a, i_h] = np.log10(np.abs(euler_raw2 / c) + 1e-16)

    return euler

def timing(solver, cp, rep=4, do_print=False):

    ug_grid_all = UCGrid((cp.b, cp.grid_max_A, cp.grid_size),
                         (cp.b, cp.grid_max_H, len(cp.asset_grid_H)))

    time_best = np.inf

    # ls MODEL
    for i in range(rep):

        solution = solver(cp)
        euler = euler_housing(solution, cp)
        tot_time = solution['avg_time']

        if do_print:
            print(f'{i}: {tot_time:.2f} secs, euler: {np.nanmean(euler):.3f}')
            print(f'RMSE: {np.nanmean((model.sim.euler)**2)}')
            print(f'50th percentile: {np.nanpercentile(euler, 50)}')
            print(f'95th percentile: {np.nanpercentile(euler, 99)}')
            print(f'5th percentile: {np.nanpercentile(euler, 5)}')
            print(f'75th percentile: {np.nanpercentile(euler, 75)}')
            print(f'0.1th percentile: {np.nanpercentile(euler, 0.1)}')
            print(f'Kurtosis of Euler Errors: {kurtosis(model.sim.euler[~np.isnan(euler)], nan_policy="omit")}')

        if tot_time < time_best:
            time_best = tot_time
            model_best = solution

    model_best['euler'] = euler

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

        if t < cp.T:
            times.append(time.time() - start)

        t = t - 1

    results['avg_time'] = np.mean(times)

    return results

def solveEGM(cp, LS=True, verbose=True):
    _, iterEGM, condition_V, _ = Operator_Factory(cp)

    # Initial values
    EVnxt, ELambdaAnxt, ELambdaHnxt = initVal(cp)

    t = cp.T
    times = []
    results = {}

    while t >= cp.t0:
        start = time.time()

        (Vcurr, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, AdjPol,
         KeeperPol, EGMGrids) = iterEGM(t, EVnxt, ELambdaAnxt, ELambdaHnxt)

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
        results[t]["EGMGrids"] = EGMGrids
        results[t]["ELambdaHnxt"] = ELambdaHnxt

        if verbose:
            print(f"EGM age {t}, time is {time.time() - start}")

        t = t - 1

    results['avg_time'] = np.mean(times)

    return results

def solveNEGM(cp, LS=True, verbose=True):
    iterVFI, iterEGM, condition_V, iterNEGM = Operator_Factory(cp)

    # Initial values
    EVnxt, ELambdaAnxt, ELambdaHnxt = initVal(cp)

    # Results dictionaries
    results = {}
    times = []
    t = cp.T

    while t >= cp.t0:
        results[t] = {}
        start = time.time()

        (Vcurr, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, AdjPol, KeeperPol) = iterNEGM(
            EVnxt, ELambdaAnxt, ELambdaHnxt, t)

        if verbose:
            print(f"NEGM age {t}, time {time.time() - start}")

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

        t = t - 1

    results['avg_time'] = np.mean(times)

    return results


if __name__ == "__main__":
	
	# Read settings
	with open("../settings/settings.yml", "r") as stream:
		eggbasket_config = yaml.safe_load(stream) 

	cp = ConsumerProblem(eggbasket_config,
						 r=0.034,
						 sigma=.001,
						 r_H=0,
						 beta=.91,
						 alpha=0.12,
						 delta=0,
						 Pi=((0.2, 0.8), (0.8, 0.2)),
						 z_vals=(1, 0.25),
						 b=1e-10,
						 grid_max_A=15.0,
						 grid_max_WE=70.0,
						 grid_size_W=300,
						 grid_max_H=50.0,
						 grid_size=300,
						 grid_size_H=300,
						 gamma_c=3,
						 chi=0,
						 tau=0.18,
						 K=1.3,
						 tol_bel=1e-09,
						 m_bar=1.0001,
						 theta=np.exp(0.3), t0 =58, root_eps=1e-1, stat= False)
	
	# 0. Solve with Bellman 
	#iterVFI, iterEGM, condition_V, NEGM = Operator_Factory(cp)
	#pickle.dump(bell_results, open("bell_results_300.p", "wb"))
	#bell_results = pickle.load(open("bell_results_300.p", "rb"))

	# 1. Solve using NEGM and EGM 
	
	NEGMRes = timing(solveNEGM, cp, rep =1)
	EGMRes = timing(solveEGM, cp, rep =1)
	NEGMRes['label'] = 'NEGM'
	EGMRes['label'] = 'EGM'
 
	# 2a. Basic policy function plots
	plot_pols(cp,EGMRes, NEGMRes, 58, [100,150,200])

	# 2.b. plot of adjuster endog. grids 
	plot_grids(EGMRes,cp,term_t = 57)
		 
	# 3. Euler errors
	eulerNEGM = euler_housing(NEGMRes, cp)
	eulerEGM = euler_housing(EGMRes, cp)
	print("NEGM Euler error is {}".format(np.nanmean(eulerNEGM)))
	print("EGM Euler error is {}".format(np.nanmean(eulerEGM)))

	# 4. Tabulate timing and errors with latex table 
	print("NEGM Time is {}".format(NEGMRes['avg_time']))
	print("EGM Time is {}".format(EGMRes['avg_time']))

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
	txt += f' | {EGMRes["avg_time"]:.2f}'
	txt += f' | {NEGMRes["avg_time"]:.2f}'
	txt += ' |\n'

	lines.append(txt)
	lines.insert(0, '| Method | EGM | NEGM |\n')

	with open("table_housing.tex", "w") as f:
		f.writelines(lines)


