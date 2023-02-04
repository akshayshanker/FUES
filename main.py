"""

Author: Akshay Shanker, University of New South Wales, a.shanker@unsw.edu.au

"""
import numpy as np
from scipy.optimize import minimize, root, fsolve
import scipy.optimize as optimize
from quantecon.optimize import brentq
from quantecon import MarkovChain
import quantecon.markov as Markov

import quantecon as qe
from numba import jit, vectorize
#from pathos.pools import ProcessPool 
import time
import  dill as pickle
import yaml
 
import matplotlib.pylab as pl

from numba import njit, prange


from solvers import ConsumerProblem, Operator_Factory


# Read settings
with open("settings/settings.yml", "r") as stream:
	eggbasket_config = yaml.safe_load(stream)


# Generate class 



cp = ConsumerProblem(eggbasket_config, 
				 r = 0.024, 
				 sigma = .01,
				 r_H = .03,
				 beta =.945, 
				 alpha = 0.33, 
				 delta = 0,
				 Pi=((0.1, 0.8, 0.1), (.4, 0.4,0.2),(.2, 0.4,0.4)),
				 z_vals = (.1, 1, 1.1), 
				 b = .01, 
				 grid_max_A = 50.0, 
				 grid_max_WE = 1000.0,
				 grid_max_H = 50.0, 
				 grid_size = 250,
				 gamma_c = 3.5,
				 chi = 0,
				 tau = 0.15, 
				 K = 20,
				 theta = 2)

bellman_operator, coleman_operator, condition_V = Operator_Factory(cp)

# === Solve r.h.s. of Bellman equation === #
shape = (len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
shape_all = (60, len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
V_init, h_init, a_init  = np.empty(shape), np.empty(shape), np.empty(shape)

V_init, Ud_prime_a_init, Ud_prime_h_init  = np.empty(shape), np.empty(shape), np.empty(shape)
for state in prange(len(cp.X_all)):
	a = cp.asset_grid_A[cp.X_all[state][1]]
	h = cp.asset_grid_H[cp.X_all[state][2]]
	i_a = cp.X_all[state][1]
	i_h = cp.X_all[state][2]
	i_z = cp.X_all[state][0]
	z   = cp.z_vals[i_z]


	V_init[i_z, i_a, i_h] = cp.term_u(cp.R_H*(1-cp.delta)*h + cp.R*a)
	Ud_prime_a_init[i_z, i_a, i_h] = cp.term_du(cp.R_H*(1-cp.delta)*h + cp.R*a)
	Ud_prime_h_init[i_z, i_a, i_h] = cp.term_du(cp.R_H*(1-cp.delta)*h + cp.R*a)

V_pol_vfi, h_pols_vfi, a_pols_vfi = np.empty(shape_all),  np.empty(shape_all),  np.empty(shape_all)
V_pols, h_pols, a_pols, c_pol = np.empty(shape_all),  np.empty(shape_all),  np.empty(shape_all), np.empty(shape_all)
a_pols_adj = np.empty(shape_all)
h_pols_adj = np.empty(shape_all)

t = cp.T
V = np.copy(V_init)
while t>50: 
	a_pols_new, h_pols_new, V_pols_new, new_c, a_adj_pol, h_adj_pol = bellman_operator(t,V)
	new_V, new_UD_a,new_UD_h = condition_V(V_pols_new, V_pols_new,V_pols_new)
	V = np.copy(new_V)
	a_pols[int(t-1)], h_pols[int(t-1)],V_pols[int(t-1)] = np.copy(a_pols_new), np.copy(h_pols_new), np.copy(new_V)
	a_pols_adj[int(t-1)] = np.copy(a_adj_pol)
	c_pol[int(t-1)]  = new_c
	h_pols_adj[int(t-1)] = np.copy(h_adj_pol)
	print(t)
	t = t-1

# Solve using Coleman

V = np.copy(V_init)
Ud_prime_a = Ud_prime_a_init 
Ud_prime_h = Ud_prime_h_init 

V_pol_col, h_pols_col, a_pols_col = np.empty(shape_all),  np.empty(shape_all),  np.empty(shape_all)
c_pol_col = np.empty(shape_all)
t = cp.T

"""
while t>50: 

	new_a_prime, new_h_prime, new_V_uc, new_UD_a_uc,new_UD_h_uc, new_c = coleman_operator(t,V,Ud_prime_a,Ud_prime_h)
	new_V, new_UD_a,new_UD_h = condition_V(new_V_uc, new_UD_a_uc,new_UD_h_uc)
	Ud_prime_a = np.copy(new_UD_a)
	Ud_prime_h = np.copy(new_UD_h)
	V = np.copy(new_V)
	V_pol_col[int(t-1)] = np.copy(new_V)
	a_pols_col[int(t-1)] = np.copy(new_a_prime)
	h_pols_col[int(t-1)] = np.copy(new_h_prime)
	c_pol_col[int(t-1)] = np.copy(new_c)
	
	print(t)
	t = t-1
pl.close()
fig, ax = pl.subplots(1,2)
colors = pl.cm.jet(np.linspace(0,1,cp.grid_size))

plot_t = 53

"""

pl.close()

fig, ax = pl.subplots(1,2)
for i_h in [3,10,20,25,40]:
	g_h = h_pols_adj[plot_t,1,:, i_h] 
	g_a = a_pols_adj[plot_t,1,:, i_h] 
	g_h_col = h_pols_col[plot_t,1,:, i_h] 
	g_a_col = a_pols[plot_t,1,:, i_h] 

	#v_bell = V_pols[plot_t,1, i_h,:]
	#sv_col = V_pol_col[plot_t,1, i_h,: ]

	#ax[0,0].plot(cp.asset_grid_H,g_h,color=colors[i_h])
	#ax[0,1].plot(cp.asset_grid_H,g_h_col,color=colors[i_h])
	ax[0].plot(cp.asset_grid_H,g_a,color=colors[i_h])
	ax[1].plot(cp.asset_grid_H,g_h,color=colors[i_h])
	#ax[1].set_antialiased(True)
	#ax[2,0].plot(cp.asset_grid_H,v_bell,color=colors[i_h])
	#ax[2,1].plot(cp.asset_grid_H,v_col,color=colors[i_h])
	ax[0].set_xlabel('Assets (t)')
	ax[1].set_xlabel('Assets (t)')
	#ax[0,0].set_ylabel('H policy')
	ax[0].set_ylabel('A policy')
	ax[1].set_ylabel('A policy')
	#ax[2,1].set_xlabel('Housing assets (t)')

	#ax[1].set_title('VFI')
	#ax[0,1].set_title('CEO-EGM')
	fig.tight_layout()

fig.savefig('test_7.png') 


"""
pl.close()
fig, ax = pl.subplots(3,2)
colors = pl.cm.jet(np.linspace(0,1,cp.grid_size))
for i_a in [4,10,30,31, 32,45]:
	g_h = h_pols[plot_t,1,:, i_a] 
	g_a = c_pol[plot_t,1, :, i_a] 
	g_h_col = h_pols_col[plot_t,1,:, i_a] 
	g_a_col = c_pol_col[plot_t,1,:,i_a] 

	v_bell = V_pols[plot_t,1, :,i_a]
	v_col = V_pol_col[plot_t,1, :,i_a ]

	ax[0,0].plot(cp.asset_grid_A,g_h,color=colors[i_a])
	ax[0,1].plot(cp.asset_grid_A,g_h_col,color=colors[i_a])
	ax[1,0].plot(cp.asset_grid_A,g_a,color=colors[i_a])
	ax[1,1].plot(cp.asset_grid_A,g_a_col,color=colors[i_a])
	ax[2,0].plot(cp.asset_grid_A,v_bell,color=colors[i_a])
	ax[2,1].plot(cp.asset_grid_A,v_col,color=colors[i_a])
	ax[2,0].set_xlabel('Housing assets (t)')
	ax[0,0].set_ylabel('H policy')
	ax[1,0].set_ylabel('A policy')
	ax[2,0].set_ylabel('Value function')
	ax[2,1].set_xlabel('Housing assets (t)')

	ax[0,0].set_title('VFI')
	ax[0,1].set_title('CEO-EGM')
	fig.tight_layout()

fig.savefig('test_6.png') 
"""
