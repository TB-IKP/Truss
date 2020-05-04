import numpy as np
import numdifftools as nd

from .plot import *
from .solve import *
from .functions import *
from .auxiliary_truss import *
from .auxiliary_solve import *

#---------------------------------------------------------------------------------------#
#		Class
#---------------------------------------------------------------------------------------#

class Truss:

	def __init__(self,nodes,fixed_nodes,load_cases,bars=None,max_length=1e6,start_diameter=0,
			young_E=1,min_diameter=0,max_diameter=100,max_compliance=100,max_stress=1,
			verbose=True):

		self.nodes 		= nodes
		self.fixed_nodes	= fixed_nodes
		self.free_nodes		= np.setdiff1d(range(nodes.shape[0]),fixed_nodes)
		self.load_cases 	= load_cases

		self.par     		= {'E': 		float(young_E),
					   'max_length':	float(max_length),
					   'min_diam':		float(min_diameter),
					   'max_diam':		float(max_diameter),
					   'max_comp':		float(max_compliance),
					   'max_stress':	float(max_stress),
					   'dim':		self.nodes.shape[1],
					   'n_n':		self.nodes.shape[0],
					   'num_fixed':		len(self.fixed_nodes),
					   'n_fn':		self.nodes.shape[0]-len(self.fixed_nodes),
					   'n_lc':		self.load_cases.shape[-1],
					   }
		
		if isinstance(bars,np.ndarray):
			self.bars = bars
		else:
			self.bars = potential_bars(self)

		#additional parameters
		self.par['n_b'] 	= len(self.bars)
		self.par['n_dl'] 	= self.par['n_fn'] * self.par['dim'] * self.par['n_lc']
		self.par['n_var'] 	= self.par['n_b'] + self.par['n_dl']

		self.bar_diam 		= np.ones((self.par['n_b']))*start_diameter
		self.bar_lengths 	= bar_lengths(self)
		self.bar_angles  	= bar_angles(self)

		#collect additional options for ipopt
		self.options_ipopt 	= []

		#initialize parameters for ALM
		self.method_ALM 	= False

		#print output
		self.verbose 		= verbose

	#objective function and gradient

	def objective(self,x):

		out_obj = objective(self,x)

		if self.method_ALM:

			out_obj += augmented_lagrangian(self,x)

		return out_obj

	def gradient(self,x):

		return nd.Gradient(self.objective)(x)

	#constraints

	def linear(self):
		return linear(self)

	def nonlinear(self,x):
		return nonlinear(self,x)

	def vanishing(self,x):

		if self.method_ALM:

			#if the ALM is used, the vaishing constraints GH are relaxed
			#and H>=0 is absorbed in the bounds

			out_van 	= np.concatenate((van_GH_diam(self,x),
							van_GH_stress(self,x)
							))

		else:
			out_van 	= np.concatenate((#van_H(self,x),
							van_GH_diam(self,x),
							van_GH_stress(self,x)
							))

		return out_van#,out_jac_van

	#limits for constraints

	def limits(self):

		out_limits_lower = np.concatenate((linear_limits(self)[0],
						nonlinear_limits(self)[0]
						))
		out_limits_upper = np.concatenate((linear_limits(self)[1],
						nonlinear_limits(self)[1]
						))

		if not self.method_ALM:

			out_limits_lower = np.concatenate((out_limits_lower,vanishing_limits(self)[0]))
			out_limits_upper = np.concatenate((out_limits_upper,vanishing_limits(self)[1]))

		return out_limits_lower,out_limits_upper

	#bounds

	def bounds(self):

		out_bounds_lower = np.concatenate((np.zeros((self.par['n_b'])),
						np.ones((self.par['n_dl']))*(-1e19)
						))
		out_bounds_upper = np.concatenate((np.ones((self.par['n_b']))*self.par['max_diam'],
						np.ones((self.par['n_dl']))*(1e19)
						))

		return out_bounds_lower,out_bounds_upper

	#ipopt

	def constraints(self,x):

		out_constr 	= np.concatenate([np.dot(self.linear(),x),
						self.nonlinear(x)
						])

		if not self.method_ALM:

			out_constr = np.concatenate((out_constr,self.vanishing(x)))

		return out_constr

	def jacobian(self,x):

		out_jac 	= np.concatenate([self.linear(),
						nd.Jacobian(self.nonlinear)(x)
						]).flatten()

		if not self.method_ALM:

			out_jac = np.concatenate((out_jac,nd.Jacobian(self.vanishing)(x).flatten()))

		return out_jac

	#collect additional options for ipopt

	def add_option(self,*args):
		
		self.options_ipopt.append(*args)

	#solve

	def solve(self,method,**kwargs):

		if method == 'direct':
			return solve_direct(self)
		
		elif method in ['ALM','alm']:

			self.par_ALM 	= {'n':			self.par['n_var'],
					   'x':			np.zeros((self.par['n_var'])),
					   'eta':		np.zeros((self.par['n_b']*(self.par['n_lc']+1))),
					   'eta_max':		1e4,
					   'alpha':		1.0,
					   'gamma':		2.0,
					   'tau':		0.1,
					   'iter':		0,
					   'max_iter':		200,
					   'stop_crit':		1e-6,
					   }

			for key in kwargs:

				if key in ['x0','eta0']:
					self.par_ALM[key[:-1]] 	= kwargs[key].astype(float)

				elif key in ['eta_max','alpha','gamma',
						'tau','max_iter','stop_crit']:
					self.par_ALM[key] 	= float(kwargs[key])

				else:
					raise KeyError('key %s not known!'% key)

			return solve_alm(self)
		
		else:
			raise ValueError('method must be direct or ALM!')

	#model data
	def stress(self,x):
		return stress(self,x)

	def volume(self,x):
		return objective(self,x)

	#plots

	def plot_initial(self):
		plot_initial(self)

	def plot_optimal(self,x):
		plot_optimal(self,x)

	def plot_loaded(self,x,color=False):
		plot_loaded(self,x,color)








