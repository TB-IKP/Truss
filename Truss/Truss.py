'''Optimization of truss structures using the Augmented Lagrangian Method'''

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
	'''Class for definition of trusses in the ground-structure approach
	and subsequent optimization using the Augmented Lagrangian Method.

	The trusses are modelled as mathematical programs with multi-dimensional vanishing constraints:

	min sum_i l_i*x_i  s.t.	K(x)u 		 		 = F_{ext} 
				F_{ext}^T*u 			<= c
				x_i				>= 0
				x_i				<= x_{max}
				(x_{min}-x_i)*x_i 		<= 0
				(sigma_i^2-sigma_{max}^2)*x_i 	<= 0

	Attributes:
	----------
	nodes: array
		information on nodal coordinates, e.g. [[x0,y0],[x1,y1],[x2,y2],...]
	fixed_nodes: list
		indices of fixed nodes with respect to nodes input
	load_cases: array
		information on applied loads; [#nodes,dimension,#loads]
	bars: array
		indices of start and end nodes, e.g. [[0,1],[1,2]]
		for bars between nodes 0 and 1 and nodes 1 and 2
		If no input is provided, all possible bars will be selected.
	
	max_length: float, default = 1e6
		maximum bar length
	start_diameter: float, default 0
		initial bar diameter the optimization process is started with
	young_E: float, default = 1
		Young's modulus; equal for all bars
	min_diameter: float, default = 0
		minimum required bar diameter after optimization
	max_length: float, default = 100
		maximum required bar diameter after optimization
	max_compliance: float, default = 10
		constraint on compliance c in F_{ext}^T*u <= c
	max_stress: float, default = 1
		maximum stress sigma_{max} on single realized bar after optimization
	verbose: bool, default = True
		Status output is printed if True.


	Methods:
	--------
	add_option(*args)
		Pass additional options to Ipopt.
	solve(method,**kwargs)
		Find an optimal structure for the given truss 
		subject to the defined constraints and load cases.
		Method can be either Ipopt or ALM.
		Additional ALM-specific parameters can be passed as keyword arguments.

	stress(x):
		Determine stress on the individual bars.
	volume(x):
		Return the total volume of the structure.

	plot_initial():
		Plot the initial ground structure.
	plot_optimal(x):
		Plot the optimal solution without considering nodal displacements.
	plot_loaded(x,color)
		Plot the optimal solution considering nodal displacements.
		Putting color=True, the bars are color-coded in the stress value.

	See the docstrings of the individual methods for variable meanings.
	'''

	def __init__(self,nodes,fixed_nodes,load_cases,bars=None,max_length=1e6,start_diameter=0,
			young_E=1,min_diameter=0,max_diameter=100,max_compliance=10,max_stress=1,
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
		'''Calculate the value of the objective function.
		If the ALM method is chosen, this is the Augmented Lagrangian.'''

		out_obj = objective(self,x)

		if self.method_ALM:

			out_obj += augmented_lagrangian(self,x)

		return out_obj

	def gradient(self,x):
		'''Calculate the gradient of the objective function.'''

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

		return out_van

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
		'''Pass additional options to Ipopt.'''

		self.options_ipopt.append(*args)

	#solve

	def solve(self,method,**kwargs):

		if method in ['Ipopt','IPOPT','ipopt','direct']:
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
		'''Determine stress on the individual bars.

		Parameters:
		-----------
		x: array
			Bar diameters and nodal displacements as obtained from solve() method.
		'''

		return stress(self,x)

	def volume(self,x):
		'''Return the total volume of the structure.

		Parameters:
		-----------
		x: array
			Bar diameters and nodal displacements as obtained from solve() method.
		'''

		return objective(self,x)

	#plots

	def plot_initial(self):
		'''Plot the initial ground structure.'''

		plot_initial(self)

	def plot_optimal(self,x):
		'''Plot the optimal solution without considering nodal displacements.

		Parameters:
		-----------
		x: array
			Bar diameters and nodal displacements as obtained from solve() method.
		'''

		plot_optimal(self,x)

	def plot_loaded(self,x,color=False):
		'''Plot the optimal solution considering nodal displacements.
		Putting color=True, the bars are color-coded in the stress value.

		Parameters:
		-----------
		x: array
			Bar diameters and nodal displacements as obtained from solve() method.
		color: bool, default = False
			Defines whether the bars are color-coded in the stress value.
		'''

		plot_loaded(self,x,color)








