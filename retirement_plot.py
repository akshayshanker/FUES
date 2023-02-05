"""Solution of Ishkakov et al (2017) retirement choice
model using FUES-EGM by Dobrescu and Shanker (2022).

Author: Akshay Shanker, University of Sydney, akshay.shanker@me.com.

Todo
---

1. Implement forward scan 

2. Error comparision with interpolated policy function 
	rather than value function
	if DC-EGM and FUES are not exact

"""

import numpy as np
from numba import jit
import time
import dill as pickle
from numba import njit, prange

from FUES.FUES import FUES

from FUES.math_funcs import interp_as, upper_envelope

from HARK.interpolation import LinearInterp
from HARK.dcegm import calc_segments, calc_multiline_envelope,\
						 calc_cross_points
from interpolation import interp

import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl



def plot_egrids(age, e_grid, vf_work,sigma_work_dt, a_prime):
	# Plot unrefined vs refined endogenous grid for different ages

	# get unrefined endogenous grid, value function and consumption 
	# for worker at time t
	x = np.array(e_grid[age])
	vf = np.array(vf_work[age])
	c = np.array(sigma_work_dt[age])
	a_prime = np.array(cp.asset_grid_A)

	# generate refined grid, value function and policy using FUES 
	x_clean, vf_clean, c_clean,a_prime_clean, dela \
						= FUES(x, vf, c, a_prime,0.8)
	# interpolate
	vf_interp_fues  = np.interp(x,x_clean, vf_clean)

	# Plot  upper envelope and asset policy 
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
		edgecolors='r')
	ax[0].plot(
		x_clean,
		vf_clean * cp.beta - cp.delta,
		color='black',
		linewidth=1,
		label='Value function')
	ax[0].scatter(
		x_clean,
		vf_clean * cp.beta - cp.delta,
		color='blue',
		s=15,
		marker='x',
		linewidth=0.75)

	#ax[0].set_ylim(7.75, 8.27)
	#ax[0].set_xlim(44, 54.5)
	#ax[0].set_ylim(7.75,8.27)
	#ax[0].set_xlim(48,56)
	ax[0].set_xlabel('Assets (t)', fontsize=11)
	ax[0].set_ylabel('Value', fontsize=11)
	ax[0].spines['right'].set_visible(False)
	ax[0].spines['top'].set_visible(False)
	ax[0].legend(frameon=False, prop={'size': 10})
	ax[0].set_yticklabels(ax[0].get_yticks(), size=9)
	ax[0].set_xticklabels(ax[0].get_xticks(), size=9)
	ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
	ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

	ax[1].scatter(
		np.sort(x),
		np.take(
			x  - c,
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
		label='Optimal points')

	ax[1].set_ylim(20,45)
	ax[1].set_xlim(44, 54.2)
	#ax[1].set_ylim(20,55)
	#ax[1].set_xlim(48,60)
	ax[1].set_ylabel('Assets (t+1)', fontsize=11)
	ax[1].set_xlabel('Assets (t)', fontsize=11)
	ax[1].spines['right'].set_visible(False)
	ax[1].spines['top'].set_visible(False)
	ax[1].set_yticklabels(ax[1].get_yticks(), size=9)
	ax[1].set_xticklabels(ax[1].get_xticks(), size=9)
	ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
	ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
	fig.tight_layout()
	ax[1].legend(frameon=False, prop={'size': 10})
	fig.savefig('plots/retirement/ret_vf_aprime_all_{}_{}.png'.format(age, g_size))
	pl.close()

	return None


def plot_cons_pol(e_grid, vf_work,sigma_work, a_prime):
# Plot consumption policy  for difference ages 
	sns.set(style="whitegrid",
			rc={"font.size": 10,
				"axes.titlesize": 10,
				"axes.labelsize": 10})
	fig, ax = pl.subplots(1, 1)

	for t, col, lab in zip([17, 10, 0], ['blue', 'red', 'black'], [
						   't=18', 't=10', 't=1']):

		cons_pol = np.copy(sigma_work[t])

		# remove jump joints 
		pos = np.where(np.abs(np.diff(cons_pol)/np.diff(cp.asset_grid_A)) > 0.4)[0] + 1
		y1 = np.insert(cons_pol, pos, np.nan)
		x1 = np.insert(cp.asset_grid_A, pos, np.nan)

		ax.plot(x1, y1, color=col, label=lab)
		ax.set_xlim(0, 380)
		ax.set_ylim(0, 40)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_yticklabels(ax.get_yticks(), size=9)
		ax.set_xticklabels(ax.get_xticks(), size=9)
		ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
		ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
		ax.set_ylabel('Consumption', fontsize=11)
		ax.set_xlabel('Assets (t)', fontsize=11)

	ax.legend(frameon=False, prop={'size': 10})
	fig.savefig('plots/retirement/ret_cons_all.png'.format(t))
	pl.close()
	
	return None

def plot_dcegm_cf(age, g_size, e_grid, vf_work,sigma_work_dt, a_prime,\
					 plot = True):
	# get unrefined endogenous grid, value function and consumption 
	# for worker at time t
	x = e_grid[age]
	vf = vf_work[age]
	c = sigma_work_dt[age]
	a_prime = cp.asset_grid_A
	time_start_dcegm = time.time()
	

	start, end = calc_segments(x, vf)

	# generate refined grid, value function and policy using FUES 
	x_clean, vf_clean, c_clean,a_prime_clean, dela = FUES(x, vf,\
														 c, a_prime,m_bar =3)
	#print(np.where(np.array(x) == x_clean))
	# interpolate
	vf_interp_fues  = np.interp(x,x_clean, vf_clean)
	#len(vf_interp_fues[x_clean.searchsorted(x)])
	vf_interp_fues[x.searchsorted(x_clean)] = vf_clean



	# Plot them, and store them as [m, v] pairs
	segments = []
	c_segments = []
	a_segments = []
	m_segments = []
	v_segments = []

	for j in range(len(start)):
		idx = range(start[j],end[j]+1)
		#pl.plot(x[idx], vf[idx])
		segments.append([x[idx], vf[idx]])
		c_segments.append(c[idx])
		a_segments.append(a_prime[idx])
		m_segments.append(x[idx])
		v_segments.append(vf[idx])


	m_upper, v_upper, inds_upper = upper_envelope(segments)
	vf_interp_fues  = np.interp(m_upper,x_clean, vf_clean)
	a_interp_fues = np.interp(m_upper,x_clean, a_prime_clean)
	#print(np.mean(np.abs()))

	c1_env = np.zeros_like(m_upper) + np.nan
	a1_env = np.zeros_like(m_upper) + np.nan
	v1_env = np.zeros_like(m_upper) + np.nan
	#print(len(inds_upper))
	
	for k, c_segm in enumerate(c_segments):
		#print(len(c_segm))
		c1_env[inds_upper == k] = c_segm[m_segments[k]\
							.searchsorted(m_upper[inds_upper == k])]

	for k, a_segm in enumerate(a_segments):
		a1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k],\
							m_segments[k], a_segm)

	for k, v_segm in enumerate(v_segments):
		v1_env[inds_upper == k] = LinearInterp(m_segments[k], v_segm)\
							(m_upper[inds_upper == k])

	a1_up = LinearInterp(m_upper, a1_env)
	indices = np.where(np.in1d(a1_env, a_prime))[0]
	a1_env2 = a1_env[indices]
	m_upper2 = m_upper[indices]

	if plot== True:

		pl.close()
		fig, ax = pl.subplots(1, 2)
		sns.set(
		style="whitegrid", rc={
		"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

		ax[1].scatter(
		x,
		vf * cp.beta - cp.delta,
		s=20,
		facecolors='none',
		label = 'EGM points',
		edgecolors='r')


		ax[1].scatter(
		x_clean,
		vf_clean * cp.beta - cp.delta,
		color='blue',
		s=15,
		marker='o',
		label = 'FUES-EGM optimal points',
		linewidth=0.75)


		ax[0].scatter(
				x,
				a_prime,
				color='r',
				s=15,
				label = 'EGM point',
				linewidth=0.75)


		ax[0].scatter(m_upper2,a1_env2,
		color='red',
		marker='x',
		label = 'DC-EGM policy function',
		linewidth=0.75)

		for k, a_segm in enumerate(a_segments):
			ax[0].plot(m_segments[k],a_segm,
			color='black',
			linestyle='--',
			#label = 'DC-EGM policy function',
			linewidth=0.75)
			#print(m_segments[k])
		
		for k, v_segm in enumerate(v_segments):
			ax[1].plot(m_segments[k],v_segm* cp.beta - cp.delta,
			color='black',
			linestyle='--',
			linewidth=0.75)

	#print(a_segm)

		ax[0].scatter(
				x_clean,
				a_prime_clean,
				color='blue',
				s=15,
				marker='x',
				label = 'FUES-EGM optimal points',
				linewidth=0.75)


		ax[1].set_ylim(6,6.4)
		ax[1].set_xlim(20,30)
		ax[1].set_xlabel('Assets (t)', fontsize=11)
		ax[1].set_ylabel('Value', fontsize=11)
		ax[1].spines['right'].set_visible(False)
		ax[1].spines['top'].set_visible(False)
		ax[1].legend(frameon=False, prop={'size': 10})


		ax[0].set_ylim(0,5)
		ax[0].set_xlim(20,40)
		ax[0].set_xlabel('Assets (t)', fontsize=11)
		ax[0].set_ylabel('Assets (t+1)', fontsize=11)
		ax[0].spines['right'].set_visible(False)
		ax[0].spines['top'].set_visible(False)
		ax[0].legend(frameon=False, prop={'size': 10})


		fig.tight_layout()
		fig.savefig('plots/retirement/ret_vf_aprime_all_{}_cf_{}.png'\
				.format(g_size,age))

	return v_upper,v1_env, vf_interp_fues,a_prime_clean, m_upper, a1_env2



if __name__ == "__main__":

	from examples.retirement_choice import Operator_Factory,RetirementModel
	
	g_size = 2000
	beta_min = 0.85
	beta_max = 0.98
	N_params = 100

	# age at which to compcare DC-EGM with FUES
	age_dcegm = 15

	errors = np.empty(N_params)

	# index for parameter draw
	param_i = 0

	# Compare values policy from DC-EGM with FUES 
	# Note we solve the model using FUES. Then at age_dcegm, we take the full
	# EGM grid and compute the upper envelope using DC-EGM and compare to FUES. 
	# Comparison performed on EGM grid points selected by DC-EGM (not all EGM
	# points, to avoid picking up interpolation error)

	
	for beta in np.linspace(beta_min,beta_max,N_params):
		
		# Create instance of RetirementModel 
		cp = RetirementModel(r=0.02,
							 beta=beta,
							 delta=1,
							 y=20,
							 b=1E-1,
							 grid_max_A=500,
							 grid_size=g_size,
							 T=20, 
							 smooth_sigma = 0)
		
		# Unpack solvers
		Ts_ret, Ts_work, iter_bell = Operator_Factory(cp)

		# Get optimal value and policy functions using FUES
		e_grid,vf_work,vf_uncond, sigma_work_dt,sigma_work = iter_bell(cp)


		# calc upper envelope using DC-EGM and compare on EGM points to 
		# FUES 
		v_upper, v1_env,vf_interp_fues,a_interp_fues,m_upper, a1_env \
			= plot_dcegm_cf(age_dcegm, g_size, e_grid,\
				 			vf_work,sigma_work_dt, cp.asset_grid_A,\
				 			plot = True)

		if len(a1_env) == len(a_interp_fues):
			errors[param_i] = \
					np.max(np.abs(a1_env-a_interp_fues))/len(a1_env)

		else: 
			errors[param_i] =\
					 np.max(np.abs(vf_interp_fues-v_upper))/len(v_upper)

		print(errors[param_i ])
		print(beta)
		
		param_i = param_i +1

	print("Avg error between DC-EGM and FUES for {} beta values between {}  and {} is {}".format(N_params, beta_min, beta_max, np.mean(errors)))


	
