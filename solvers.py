"""

Author: Akshay Shanker, University of New South Wales, a.shanker@unsw.edu.au

"""
import numpy as np
import quantecon.markov as Markov
import quantecon as qe
from numba import jit, vectorize, prange
import time
import  dill as pickle
from sklearn.utils.extmath import cartesian 
from numba import njit, prange
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interpolation.splines import extrap_options as xto
from interpolation import interp 
from quantecon.optimize.root_finding import brentq
import scipy
import matplotlib.pylab as pl


class ConsumerProblem:
	"""
	A class that stores primitives for the Consumer Problem for 
	model with fixed adjustment cost. The
	income process is assumed to be a finite state Markov chain.

	Parameters
	----------
	r : scalar(float), optional(default=0.01)
		A strictly positive scalar giving the interest rate
	Lambda: scalar(float), optional(default = 0.1)
		The shadow social value of accumulation 
	beta : scalar(float), optional(default=0.96)
		The discount factor, must satisfy (1 + r) * beta < 1
	Pi : array_like(float), optional(default=((0.60, 0.40),(0.05, 0.95))
		A 2D NumPy array giving the Markov matrix for {z_t}
	z_vals : array_like(float), optional(default=(0.5, 0.95))
		The state space of {z_t}
	b : scalar(float), optional(default=0)
		The borrowing constraint
	grid_max : scalar(float), optional(default=16)
		Max of the grid used to solve the problem
	grid_size : scalar(int), optional(default=50)
		Number of grid points to solve problem, a grid on [-b, grid_max]
	u : callable, optional(default=np.log)
		The utility function
	du : callable, optional(default=lambda x: 1/x)
		The derivative of u

	Attributes
	----------
	r, beta, Pi, z_vals, b, u, du : see Parameters
	asset_grid : np.ndarray
		One dimensional grid for assets

	"""

	def __init__(self, 
				 config, 
				 r = 0.074, 
				 sigma =1, 
				 r_H = .1,
				 beta =.945, 
				 alpha = 0.66,
				 delta = 0.1,
				 Pi = ((0.09, 0.91), (0.06, 0.94)),
				 z_vals = (0.1, 1.0), 
				 b = 1e-2, 
				 grid_max_A = 50, 
				 grid_max_WE = 100,
				 grid_max_H = 50, 
				 grid_size = 50,
				 gamma_c = 1.458,
				 K = 200,
				 theta = 2, 
				 tau = 0.2,
				 chi = 0, 
				 T = 60):
		self.grid_size = int(grid_size)
		self.r, self.R = r, 1 + r
		self.r_H, self.R_H = r_H, 1 + r_H
		self.beta = beta
		self.delta = delta 
		self.gamma_c, self.chi = gamma_c, chi
		self.b = b 
		self.T = T
		self.grid_max_A, self.grid_max_H = grid_max_A, grid_max_H
		self.sigma  = sigma
		lambdas = np.array(config['lambdas']) 
		self.alpha = alpha 

		self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)
		
		self.asset_grid_A = np.linspace(b, grid_max_A, grid_size)
		self.asset_grid_H = np.linspace(b, grid_max_H, grid_size)
		self.asset_grid_WE = np.linspace(b, grid_max_WE, grid_size)
		
		self.X_all = cartesian([np.arange(len(z_vals)),\
							np.arange(len(self.asset_grid_A)),\
							np.arange(len(self.asset_grid_H))])
		
		self.UGgrid_all = UCGrid((b, grid_max_A, grid_size),
								  (b, grid_max_H, grid_size))

		self.tau = tau 


		# define functions 
		@njit
		def du(x):
			if x<=0:
				return np.inf
			else:
				return np.power(x, - gamma_c) 
		
		@njit
		def term_du(x):
			return theta*np.power(K + x, - gamma_c) 

		@njit
		def term_u(x):
			 return theta*(np.power(K+ x,1-gamma_c)-1)/(1-gamma_c)
	
		@njit
		def u(x, y, chi):
			if x<=0:
				cons_u =  0 
			else:
				cons_u = (np.power(x,1-gamma_c)-1)/(1-gamma_c) + alpha*np.log(y)

			return cons_u - chi
		
		@njit
		def y_func(t,xi):

			t 

			wage_age = np.dot(np.array([1, t, np.power(t,2), np.power(t,3), np.power(t,4)]).astype(np.float64), lambdas[0:5])
			wage_tenure = t*lambdas[5] + np.power(t, 2)*lambdas[6] 

			return np.exp(wage_age + wage_tenure + xi)*1e-5

		self.u, self.du, self.term_u, self.term_du, self.y_func = u, du,term_u,term_du, y_func


def Operator_Factory(cp):

	# tolerances
	tol_bell = 10e-4

	beta, delta = cp.beta, cp.delta
	asset_grid_A,asset_grid_H, z_vals, Pi  = cp.asset_grid_A, cp.asset_grid_H,\
												cp.z_vals, cp.Pi
	grid_max_A, grid_max_H = cp.grid_max_A, cp.grid_max_H
	u, du, term_u  = cp.u, cp.du, cp.term_u
	y_func = cp.y_func
	
	R, R_H = cp.R, cp.R_H
	X_all = cp.X_all
	b = cp.b
	T = cp.T
	chi = cp.chi
	sigma = cp.sigma
	tau = cp.tau

	z_idx = np.arange(len(z_vals))
	
	shape = (len(z_vals), len(asset_grid_A), len(asset_grid_H))
	V_init, h_init, c_init  = np.empty(shape), np.empty(shape), np.empty(shape)
	UGgrid_all = cp.UGgrid_all

	@njit
	def obj_noadj(a_prime,a,h,z, i_z,V,R,R_H,t):
		# objective function to be *maximised* for non-adjusters
		if R * a +  y_func(t,z) - a_prime[0] >0 and a_prime[0]>b:
			h_prime = R_H*(1-delta)*h
			point = np.array([a_prime[0],h_prime])
			#if t > T-1: 
			#	Ev_prime = term_u(h_prime*R_H*(1-delta) + a_prime[0]*R)
			#else:
			Ev_prime = eval_linear(UGgrid_all, V[i_z],point)
			consumption = R*a + y_func(t,z) - a_prime[0]
			return u(consumption,h,0) + beta*Ev_prime
		else:
			return -np.inf

	@njit
	def obj_adj(x_prime,a,h,z, i_z,V,R,R_H,t):
		 # objective function to be *maximised* for adjusters
		h_prime = x_prime[1]
		if R * a + R_H*h*(1-delta) +  y_func(t,z) - x_prime[0] - x_prime[1] - tau*h_prime  >0 and x_prime[0] >b and x_prime[1] >b :
			h_prime = x_prime[1]
			a_prime = x_prime[0]
			point = np.array([a_prime,h_prime])
			#if t > T-1:  
			#	Ev_prime = term_u(h_prime*R_H*(1-delta) + a_prime*R)
			#else:
			Ev_prime = eval_linear(UGgrid_all, V[i_z],point)
			
			consumption =  R*a + R_H*h*(1-delta) +  y_func(t,z) - h_prime - a_prime - tau*h_prime
			
			return   u(consumption, h_prime,chi)+beta*Ev_prime
		else:
			return   -np.inf

	@njit(parallel = True)
	def bellman_operator(t,V):
		"""
		The approximate Bellman operator, which computes and returns the
		updated value function TV (or the V-greedy policy c if
		return_policy is True).

		Parameters
		----------
		V : array_like(float)
			A NumPy array of dim len(cp.asset_grid) times len(cp.z_vals)
		cp : ConsumerProblem
			An instance of ConsumerProblem that stores primitives
		return_policy : bool, optional(default=False)
			Indicates whether to return the greed policy given V or the
			updated value function TV.  Default is TV.

		Returns
		-------
		array_like(float)
			Returns either the greed policy given V or the updated value
			function TV.

		"""

		# === Linear interpolation of V along the asset grid === #
		#vf = lambda a, i_z: np.interp(a, asset_grid, V[:, i_z])
		

		# === Solve r.h.s. of Bellman equation === #
		new_V = np.empty(V.shape)
		new_h_prime = np.empty(V.shape) # next period capital 
		new_c= np.empty(V.shape)
		new_a_prime = np.empty(V.shape) #lisure
		new_V_adj = np.empty(V.shape)
		new_V_noadj = np.empty(V.shape)
		new_a_prime_adj = np.empty(V.shape)
		new_h_prime_adj = np.empty(V.shape)

		
		for state in prange(len(X_all)):
			a = asset_grid_A[X_all[state][1]]
			h = asset_grid_H[X_all[state][2]]
			i_a = X_all[state][1]
			i_h = X_all[state][2]
			i_z = X_all[state][0]
			z   = z_vals[i_z]

			a_max_nadj = R*a +  y_func(t,z)
			a_max_adj = R*a + R_H*h*(1-delta) +  y_func(t,z)
			
			bnds_adj = np.array([[b, asset_grid_A[-1]],[b, asset_grid_H[-1]]])
			x0_adj = np.array([b,b])
			x0_noadj = np.array([b])
			args_adj = (a, h, z, i_z, V,R, R_H,t)
			#print(obj_adj(x0_adj, a, h, z, i_z, V,R, R_H,t ))
			#obj_adj_b = - obj_adj
			#start = time.time()
			#x_prime_adj_star  = scipy.optimize.brute(obj_adj, ranges = bnds_adj, args = args_adj, finish = 'Newton',full_output=True, Ns = 100)
			#print(time.time() - start)
			x_prime_adj_star  = qe.optimize.nelder_mead(obj_adj,x0_adj,bounds= bnds_adj, args = args_adj, tol_f = 1e-12, tol_x = 1e-12)[0]
			x_prime_adj_star = x_prime_adj_star
			bnds_nadj = np.array([[b, asset_grid_A[-1]]])
			args_nadj = (a, h, z, i_z, V,R, R_H,t)
			#obj_noadj_b = - obj_noadj
			#x_prime_nadj_star  = scipy.optimize.brute(obj_noadj, ranges =  bnds_nadj, args = args_nadj, finish = 'Newton',full_output=True, Ns = 100)
			x_prime_nadj_star  = qe.optimize.nelder_mead(obj_noadj,x0_noadj,bounds= bnds_nadj, args = args_nadj, tol_f = 1e-12, tol_x = 1e-12)[0]
			#x_prime_nadj_star = np.array([x_prime_nadj_star[0]])
			consumption =  R*a + R_H*h*(1-delta) +  y_func(t,z) - x_prime_adj_star[1] - x_prime_adj_star[0] - tau*x_prime_adj_star[1]
			v_adj = obj_adj(x_prime_adj_star, a, h, z, i_z, V,R, R_H,t)
			v_nadj = obj_noadj(x_prime_nadj_star, a, h, z, i_z, V,R, R_H,t)

			d_adj = np.exp(v_adj/sigma)/(np.exp(v_adj/sigma) + np.exp(v_nadj/sigma))

			if v_adj>=v_nadj:
			#d_adj = np.exp(v_adj/sigma)/(np.exp(v_adj/sigma) + np.exp(v_nadj/sigma))
				d_adj =1
			else:
				d_adj =0
	
			v = d_adj*v_adj + (1-d_adj)*v_nadj
			h_prime = d_adj*x_prime_adj_star[1] + (1-d_adj)*R_H*h*(1-delta)
			#h_prime = d_adj
			a_prime = d_adj*x_prime_adj_star[0]+ (1-d_adj)*x_prime_nadj_star[0]

			new_h_prime[i_z,i_a, i_h],new_a_prime[i_z, i_a, i_h],new_V[i_z, i_a,i_h] = h_prime,a_prime, v
			new_c[i_z,i_a, i_h] = consumption
			new_a_prime_adj[i_z,i_a, i_h] = x_prime_adj_star[0]
			new_h_prime_adj[i_z,i_a, i_h] = x_prime_adj_star[1]
	   
		return new_a_prime, new_h_prime, new_V, new_c,new_a_prime_adj, new_h_prime_adj
	
	@njit 
	def root_H_UPRIME_func(h_prime,z,a, h, Ud_prime_a, Ud_prime_h,t):

		""" Function whose root  is solution of H to 
			 illiquid asset Euler endogenous of a_prime value 
		"""
		args = (h_prime,z,a,h, Ud_prime_a, Ud_prime_h,t)
		a_prime = brentq(root_A_UPRIME_func_adj,\
							asset_grid_A[0],\
							asset_grid_A[-1],\
							args = args,\
							disp=False, xtol = 1e-14)[0]
		
		#if R * a + R_H*h*(1-delta) +  y_func(t,z) - a_prime - h_prime < 0:
		#	return np.inf
		#else:
		c =  R * a + R_H*h*(1-delta) +  y_func(t,z) - a_prime - h_prime
		c_h_min = R * a + R_H*h*(1-delta)  +  y_func(t,z) - a_prime - asset_grid_H[0]
		c_h_max = max(0,R * a + R_H*h*(1-delta)  +  y_func(t,z) - a_prime - asset_grid_H[-1])
		
		du_val = du(c)
		point = np.array([a_prime,h_prime])

		Ud_prime_H_val = beta*R_H*(1-delta)*eval_linear(UGgrid_all, Ud_prime_h, point, xto.LINEAR)

		return du_val - min(du(c_h_max), max(Ud_prime_H_val, du(c_h_min)))

	@njit 
	def root_A_UPRIME_func_noadj(a_prime,z,a, h, Ud_prime_a, Ud_prime_h,t):
		""" Evaluate root of liquid asset Euler given value of h_prime for non-adjusters
		"""

		#if R * a  +  y_func(t,z) - a_prime < 0:
		#	return np.inf
		#else:
		c =  R * a  +  y_func(t,z) - a_prime
		c_a_min = R * a  +  y_func(t,z) - asset_grid_A[0]
		c_a_max = max(0,R * a  +  y_func(t,z) - asset_grid_A[-1])

		du_val = du(c)
		h_prime = (1-delta)*R_H*h
		point = np.array([a_prime,h_prime])

		Ud_prime_a_val = beta*R*eval_linear(UGgrid_all, Ud_prime_a, point,xto.LINEAR)

		return du_val - min(du(c_a_max), max(Ud_prime_a_val, du(c_a_min)))	

	@njit 
	def root_A_UPRIME_func_adj(a_prime,h_prime, z,a, h, Ud_prime_a, Ud_prime_h,t):
		""" Function whose root  is solution of a to 
			 liquid asset Euler given value of h_prime for adjusters
		"""

		#if R * a + R_H*h*(1-delta)  +  y_func(t,z) - a_prime - h_prime < 0:
		#	return np.inf
		#else:
		c =  R * a  + R_H*h*(1-delta) +  y_func(t,z) - a_prime - h_prime
		c_a_min = R * a  + R_H*h*(1-delta)+  y_func(t,z) - asset_grid_A[0] - h_prime
		c_a_max = max(0,R * a  + R_H*h*(1-delta)+  y_func(t,z) - asset_grid_A[-1] - h_prime)

		du_val = du(c)
		point = np.array([a_prime,h_prime])

		Ud_prime_a_val = beta*R*eval_linear(UGgrid_all, Ud_prime_a, point,xto.LINEAR)

		return du_val - min(du(c_a_max), max(Ud_prime_a_val, du(c_a_min)))


	"""
	@njit 
	def eval_func_a(t,Ud_prime_a, Ud_prime_h):
		# Evaluates a_t+1 as a function of h_t+1 conditioned 
		# on time t shock and adjusting 
		a_prime_adj_func = np.zeros((len(z_vals),len(asset_grid_H)))
		# i_z is t period shock
		for h_prime_i in range(len(asset_grid_H)):
			for i_z in range(len(z_vals)):
				h_prime = asset_grid_H[h_prime_i]
				z   = z_vals[i_z]

				point_a_lb = np.array([asset_grid_A[0],h_prime])
				point_a_ub = np.array([asset_grid_A[-1],h_prime])

				Ud_prime_a_lower_bound = R*eval_linear(UGgrid_all, Ud_prime_a[i_z], point_a_lb)
				Ud_prime_H_lower_bound = R_H*(1-delta)*eval_linear(UGgrid_all, Ud_prime_h[i_z], point_a_lb)

				Ud_prime_a_upper_bound = R*eval_linear(UGgrid_all, Ud_prime_a[i_z], point_a_ub)
				Ud_prime_H_upper_bound = R_H*(1-delta)*eval_linear(UGgrid_all, Ud_prime_h[i_z], point_a_ub)

				if Ud_prime_a_lower_bound<= Ud_prime_H_lower_bound:
					a_prime_adj_func[i_z,h_prime_i] = asset_grid_A[0]

				elif Ud_prime_a_upper_bound>= Ud_prime_H_upper_bound:
					a_prime_adj_func[i_z,h_prime_i] = asset_grid_A[-1]

				else:
					a_prime_adj_func[i_z,h_prime_i] = brentq(root_a_UPRIME_func_adj,\
															 asset_grid_A[0],\
															 asset_grid_A[-1],\
															 args = (h_prime,Ud_prime_a[i_z],Ud_prime_h[i_z]),\
															 disp=False)[0]
		return a_prime_adj_func 
		"""

	@njit 
	def coleman_operator(t,V_prime,Ud_prime_a, Ud_prime_h):

		""""

		Iterates on the Coleman operator

		Note: V_prime,Ud_prime_a, Ud_prime_h assumed 
				to be conditioned on time t shock

			 - the t+1 marginal utilities are not multiplied
			   by the discount factor and rate of return
		"""

		# Declare empty grids
		# uc indicates conditioned on time t shock
		new_a_prime = np.empty(V_prime.shape) 
		new_h_prime = np.empty(V_prime.shape)
		new_V_uc = np.empty(V_prime.shape)
		new_c = np.empty(V_prime.shape)
		new_Ud_a_uc = np.empty(V_prime.shape)
		new_Ud_h_uc = np.empty(V_prime.shape)

		# evaluate t+1 liq assets as a function of t+1 illiquid 
		# assets for adjusters 
		#a_prime_adj_func = eval_func_a(t,Ud_prime_a, Ud_prime_h)

		for state in range(len(X_all)):
			a = asset_grid_A[X_all[state][1]]
			h = asset_grid_H[X_all[state][2]]
			i_a = X_all[state][1]
			i_h = X_all[state][2]
			i_z = int(X_all[state][0])
			z   = z_vals[i_z]

			#print(z)

			#h_max_adj = (1-delta)*R_H*h + R*a+  y_func(t,z)
			#if root_H_UPRIME_func(asset_grid_H[0],z,a, h, a_prime_adj_func[i_z], Ud_prime_a[i_z], Ud_prime_h[i_z],t) >= 0:
			#	h_prime_adj = asset_grid_H[0]
			#	xi_h  = 0
			#elif root_H_UPRIME_func(asset_grid_H[-1], z,a,h, a_prime_adj_func[i_z], Ud_prime_a[i_z], Ud_prime_h[i_z],t) <= 0:
			#	h_prime_adj = asset_grid_H[-1]
			#else:
			args = (z,a,h, Ud_prime_a[i_z], Ud_prime_h[i_z],t)
			h_prime_adj = brentq(root_H_UPRIME_func,\
									asset_grid_H[0],\
									asset_grid_H[-1],\
									args = args,xtol = 1e-14,\
									disp=False)[0]

			args = (h_prime_adj,z,a,h, Ud_prime_a[i_z], Ud_prime_h[i_z],t)

			a_prime_adj = brentq(root_A_UPRIME_func_adj,\
									asset_grid_A[0],\
									asset_grid_A[-1],\
									args = args,\
									disp=False)[0]
			if h_prime_adj >= asset_grid_H[-1]:
				xi = 1
			else:
				xi =0
				
			# Eval non-adjusters 
			h_prime_nadj = (1-delta)*R_H*h
			args = (z,a,h, Ud_prime_a[i_z], Ud_prime_h[i_z],t)
			a_prime_nadj = brentq(root_A_UPRIME_func_noadj,\
												 asset_grid_A[0],\
												 asset_grid_A[-1],\
												 args = args,xtol = 1e-14,\
												 disp=False)[0]

			c_adj = max(0,R * a + R_H*h*(1-delta) +  y_func(t,z) - a_prime_adj - h_prime_adj)
			c_nadj = max(0,R * a +  y_func(t,z) - a_prime_nadj)

			point_adj = np.array([a_prime_adj, h_prime_adj])
			point_nadj = np.array([a_prime_nadj, h_prime_nadj])
			v_adj = u(c_adj, h_prime_adj,chi) + beta*eval_linear(UGgrid_all, V_prime[i_z],point_adj, xto.LINEAR) 
			v_nadj = u(c_nadj, h_prime_nadj,0) + beta*eval_linear(UGgrid_all, V_prime[i_z],point_nadj, xto.LINEAR)

			if v_adj>=v_nadj:
			#d_adj = np.exp(v_adj/sigma)/(np.exp(v_adj/sigma) + np.exp(v_nadj/sigma))
				d_adj =1
			else:
				d_adj =0

			new_h_prime[i_z, i_a, i_h] = d_adj*h_prime_adj + (1-d_adj)*h_prime_nadj
			#new_h_prime[i_z, i_a, i_h] = d_adj
			new_a_prime[i_z, i_a, i_h] = a_prime_adj*d_adj + (1-d_adj)*a_prime_nadj
			new_c[i_z, i_a, i_h] = c_adj*d_adj + (1-d_adj)*c_nadj
			new_V_uc[i_z, i_a, i_h] = d_adj*v_adj + (1-d_adj)*v_nadj
			new_Ud_a_uc[i_z, i_a, i_h] = d_adj*du(c_adj) + (1-d_adj)*du(c_nadj)
			new_Ud_h_uc[i_z, i_a, i_h] = d_adj*du(c_adj)*(1-xi) + (1-d_adj)*beta*R_H*(1-delta)*eval_linear(UGgrid_all, Ud_prime_h[i_z], point_nadj, xto.LINEAR)

		return new_a_prime, new_h_prime, new_V_uc, new_Ud_a_uc,new_Ud_h_uc,new_c

	def condition_V(new_V_uc, new_Ud_a_uc,new_Ud_h_uc):
		""" Condition the t+1 continuation vaue on 
		time t information"""

		# make the exogenuos state index the last
		#matrix_A_V = new_V_uc.transpose((1,2,0))
		#matrix_A_ua = new_Ud_a_uc.transpose((1,2,0)) 
		#matrix_A_uh = new_Ud_h_uc.transpose((1,2,0)) 

		# rows of EBA_P2 correspond to time t all exogenous state index
		# cols of EBA_P2 correspond to transition to t+1 exogenous state index
		#matrix_B = Pi

		new_V  = np.zeros(np.shape(new_V_uc))
		new_UD_a  = np.zeros(np.shape(new_Ud_a_uc))
		new_UD_h  = np.zeros(np.shape(new_Ud_h_uc))

		# numpy dot sum product over last axis of matrix_A (t+1 continuation value unconditioned)
		# see nunpy dot docs
		for state in range(len(X_all)):
			i_a = X_all[state][1]
			i_h = X_all[state][2]
			i_z = int(X_all[state][0])

			new_V[i_z, i_a, i_h] = np.dot(Pi[i_z,:], new_V_uc[:, i_a, i_h])
			new_UD_a[i_z, i_a, i_h] = np.dot(Pi[i_z,:], new_Ud_a_uc[:, i_a, i_h])
			new_UD_h[i_z, i_a, i_h] = np.dot(Pi[i_z,:], new_Ud_h_uc[:, i_a, i_h])

		return new_V, new_UD_a,new_UD_h

	return bellman_operator, coleman_operator,condition_V








