#	This file is part of Truss.
#
#	Truss is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	Truss is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with Truss.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import ipopt as ipopt
import numdifftools as nd

from .functions import *
from .auxiliary_solve import *

#---------------------------------------------------------------------------------------#
#		Direct
#---------------------------------------------------------------------------------------#

def solve_direct(self):

	#start values
	x0 = np.concatenate([np.ones((self.par['n_b']))*self.bar_diam,
				np.zeros((self.par['n_dl']))])

	#parameter bounds - bar diameters and all dislocations
	lb = self.bounds()[0]
	ub = self.bounds()[1]

	#constraints
	cl = self.limits()[0]
	cu = self.limits()[1]

	#define problem for ipopt
	problem_ipopt 	= ipopt.problem(n=self.par['n_var'],m=len(cl),problem_obj=self,
					lb=lb,ub=ub,cl=cl,cu=cu)

	#add options for ipopt
	add_option_ipopt(self,problem=problem_ipopt)

	#solve
	out_opt,out_info = problem_ipopt.solve(x0)

	return out_opt,out_info

#---------------------------------------------------------------------------------------#
#		ALM main
#---------------------------------------------------------------------------------------#

def solve_alm(self):

	self.method_ALM = True

	while self.par_ALM['iter'] < self.par_ALM['max_iter']:

		if self.verbose:
			print_stat_alm(self)

		step_ALM(self)

		if break_ALM(self):
			break

	self.method_ALM = False

	return self.par_ALM['x']

#---------------------------------------------------------------------------------------#
#		ALM step
#---------------------------------------------------------------------------------------#

def step_ALM(self):

	#Line 3: choose eta in [0,eta_max]
	self.par_ALM['eta'][self.par_ALM['eta'] > self.par_ALM['eta_max']] = self.par_ALM['eta_max']

	#Line 4: solve subproblem
	subproblem_ALM(self)

	#Line 5: update eta
	eta_new = self.par_ALM['eta'] + self.par_ALM['alpha']*self.vanishing(self.par_ALM['x'])
	eta_new[eta_new < 0] = 0

	self.par_ALM['eta'] = eta_new

	#Line 6: determine violation
	V_new = np.linalg.norm(np.min((-self.vanishing(self.par_ALM['x']),
					self.par_ALM['eta']/self.par_ALM['alpha']),axis=0))

	#Lines 7-11: evaluate progress
	if self.par_ALM['iter'] == 0 or V_new <= self.par_ALM['tau']*self.par_ALM['V']:
		self.par_ALM['alpha'] = self.par_ALM['alpha']
	else:
		self.par_ALM['alpha'] = self.par_ALM['gamma']*self.par_ALM['alpha']

	#Update parameters
	self.par_ALM['V'] 	= V_new
	self.par_ALM['iter']   += 1

	return

#---------------------------------------------------------------------------------------#
#		ALM subproblem
#---------------------------------------------------------------------------------------#

def subproblem_ALM(self):

	#parameter bounds - bar diameters and all dislocations
	lb = self.bounds()[0]
	ub = self.bounds()[1]

	#constraints
	cl = self.limits()[0]
	cu = self.limits()[1]

	#shortcut ipopt
	self.par_ALM['m'] = len(cl)

	#define problem for ipopt
	problem_ipopt 	= ipopt.problem(n=self.par_ALM['n'],m=self.par_ALM['m'],problem_obj=self,
					lb=lb,ub=ub,cl=cl,cu=cu)

	add_option_ipopt(self,problem=problem_ipopt)

	opt,info = problem_ipopt.solve(self.par_ALM['x'])
		
	self.par_ALM['x'] 		= opt
	self.par_ALM['mult_sub']	= info['mult_g']
	self.par_ALM['mult_lb']		= info['mult_x_L']
	self.par_ALM['mult_ub']		= info['mult_x_U']

	if self.verbose:
		print(info['status_msg'])
		print('x\t',self.par_ALM['x'])

	return

#---------------------------------------------------------------------------------------#
#		ALM break
#---------------------------------------------------------------------------------------#

def break_ALM(self):

	#first, the gradient of the objective function (nabla*f)
	#and the non-relaxed constraints
	#in order to obtain the non-augmented objective function, 
	#set method_ALM to False and immediately back to true

	self.method_ALM = False

	KKT_lagrangian 	= self.gradient(self.par_ALM['x'])

	self.method_ALM = True

	KKT_lagrangian += np.dot(self.par_ALM['mult_sub'].T,
				self.jacobian(self.par_ALM['x']).reshape(self.par_ALM['m'],self.par_ALM['n']))

	#the relaxed constraints (eta*nabla*GH)
	KKT_lagrangian += np.dot(self.par_ALM['eta'],nd.Jacobian(self.vanishing)(self.par_ALM['x']))

	#the bounds
	try:
		KKT_lagrangian -= self.par_ALM['mult_lb']
		KKT_lagrangian += self.par_ALM['mult_ub']

	except:
		KeyError 

	#complementarity test for relaxed constraint 
	KKT_complement = np.min((-self.vanishing(self.par_ALM['x']),self.par_ALM['eta']),axis=0)

	if self.verbose:
		print('KKT_lagrangian',np.max(np.abs(KKT_lagrangian)))
		print('KKT_complement',np.max(np.abs(KKT_complement)))
	
	if np.max(np.abs(KKT_lagrangian)) < self.par_ALM['stop_crit']:
		if np.max(np.abs(KKT_complement)) < self.par_ALM['stop_crit']:
			return True
	else:
		return False














