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
import numdifftools as nd

from .auxiliary_truss import *

#---------------------------------------------------------------------------------------#
#		Objective function and gradient
#---------------------------------------------------------------------------------------#

def objective(self,x):

	#x = [bar_diam,node_disloc]

	bar_diam 	= x[0:self.par['n_b']]
	out_objective 	= np.dot(bar_diam,self.bar_lengths)

	return out_objective

def gradient(self,x):

	out_gradient 	= nd.Gradient(objective)(x)

	return out_gradient

#---------------------------------------------------------------------------------------#
#		Augmented Lagrangian
#---------------------------------------------------------------------------------------#

def augmented_lagrangian(self,x):

	out_augmented_lagrangian = np.max((np.zeros((self.par['n_b']*(self.par['n_lc']+1))),
					self.vanishing(x) + self.par_ALM['eta']/self.par_ALM['alpha']),axis=0)
	out_augmented_lagrangian = 0.5*self.par_ALM['alpha']*np.sum(out_augmented_lagrangian**2,axis=0)

	return out_augmented_lagrangian

#---------------------------------------------------------------------------------------#
#		Linear constraints (compliance)
#---------------------------------------------------------------------------------------#

def linear(self):

	outer_forces = self.load_cases[self.free_nodes]
	outer_forces = outer_forces.reshape(self.par['n_fn']*self.par['dim'],1,self.par['n_lc']).T

	out_A = np.zeros((self.par['n_lc'],self.par['n_var']))

	for num_case in range(self.par['n_lc']):
		out_A[num_case,self.par['n_b']+num_case*self.par['n_fn']*self.par['dim']: \
			self.par['n_b']+(num_case+1)*self.par['n_fn']*self.par['dim']] = outer_forces[num_case]

	return out_A

def linear_limits(self):

	out_bl = np.ones((self.par['n_lc']))*(-1e19)
	out_bu = np.ones((self.par['n_lc']))*self.par['max_comp']

	return out_bl,out_bu

#---------------------------------------------------------------------------------------#
#		Nonlinear constraints (force equilibrium)
#---------------------------------------------------------------------------------------#

def nonlinear(self,x):

	#x = [bar_diam,node_disloc]
	bar_diam 	= x[0:self.par['n_b']]
	node_disloc 	= x[-self.par['n_dl']:].reshape(self.par['n_fn'],self.par['dim'],self.par['n_lc'])

	#stiffnes matrix
	stiff_mat 	= stiffness_matrix(self,x)

	#nonlinear constraints
	out_c  		= np.zeros((self.par['n_fn']*self.par['dim'],self.par['n_lc']))

	for num_case in range(self.par['n_lc']):
		out_c[:,num_case] = np.dot(stiff_mat,node_disloc[:,:,num_case].reshape(self.par['n_fn']*self.par['dim']))

	out_c 		= out_c.reshape(self.par['n_dl'])

	return out_c

def nonlinear_limits(self):

	outer_forces  = np.moveaxis(self.load_cases[self.free_nodes],-1,0)
	outer_forces  =	outer_forces.reshape(self.par['n_fn']*self.par['dim']*self.par['n_lc'])

	out_cl 		= outer_forces
	out_cu 		= outer_forces

	return out_cl,out_cu

#---------------------------------------------------------------------------------------#
#		Vanishing constraints (minimum diameter and stress)
#---------------------------------------------------------------------------------------#

def van_H(self,x):

	#x = [bar_diam,node_disloc]
	bar_diam 	= x[0:self.par['n_b']]
	node_disloc 	= x[-self.par['n_dl']:].reshape(self.par['n_fn'],self.par['dim'],self.par['n_lc'])

	out_H 		= np.tile(bar_diam,self.par['n_lc'])
	out_jac_H 	= np.tile(np.hstack([np.eye((self.par['n_b'])),np.zeros((self.par['n_b'],self.par['n_dl']))]),
				(self.par['n_lc'],1))

	return out_H

def van_GH_diam(self,x):

	#x = [bar_diam,node_disloc]
	bar_diam 	= x[0:self.par['n_b']]
	
	out_GH_diam 	= np.tile(self.par['min_diam'],self.par['n_b']) - bar_diam
	out_GH_diam    *= bar_diam

	out_jac_GH_diam = self.par['min_diam']*np.eye(self.par['n_b']) - 2*bar_diam*np.eye(self.par['n_b'])
	out_jac_GH_diam = np.hstack((out_jac_GH_diam,np.zeros((self.par['n_b'],self.par['n_dl']))))

	return out_GH_diam

def van_GH_stress(self,x):

	#x = [bar_diam,node_disloc]
	bar_diam 	= x[0:self.par['n_b']]
	node_disloc 	= x[-self.par['n_dl']:].reshape(self.par['n_fn'],self.par['dim'],self.par['n_lc'])

	out_GH_stress	= np.zeros((self.par['n_lc']*self.par['n_b']))
	out_jac_GH_stress = np.zeros((self.par['n_lc']*self.par['n_b'],self.par['n_var']))

	for num_case in range(self.par['n_lc']):
		for num_bar in range(self.par['n_b']):

			sigma 	= self.par['E']/self.bar_lengths[num_bar] * \
					np.dot(self.bar_angles[self.free_nodes,:,num_bar].reshape(self.par['n_fn']*self.par['dim']),
						node_disloc[:,:,num_case].reshape(self.par['n_fn']*self.par['dim']))

			out_GH_stress[num_case*self.par['n_b'] + num_bar] = (sigma**2 - self.par['max_stress']**2)*bar_diam[num_bar]

			out_jac_GH_stress[num_case*self.par['n_b']+num_bar,self.par['n_b']+num_case*self.par['n_fn']*self.par['dim']: \
					self.par['n_b']+(num_case+1)*self.par['n_fn']*self.par['dim']] = \
					out_GH_stress[num_case*self.par['n_b'] + num_bar] + \
					2*self.par['E']**2/self.bar_lengths[num_bar] * \
					np.dot(self.bar_angles[self.free_nodes,:,num_bar].reshape(self.par['n_fn']*self.par['dim']),
						node_disloc[:,:,num_case].reshape(self.par['n_fn']*self.par['dim'])) * \
					self.bar_angles[self.free_nodes,:,num_bar].reshape(self.par['n_fn']*self.par['dim'])

	return out_GH_stress

def vanishing_limits(self):

	if self.method_ALM:

		out_vl 	= []
		out_vu 	= [] 

	else:

		out_vl = np.concatenate([np.ones((self.par['n_b']))*(-1e19),
					 np.ones((self.par['n_b']*self.par['n_lc']))*(-1e19)
					]) 
		out_vu = np.concatenate([np.zeros((self.par['n_b'])),
					 np.zeros((self.par['n_b']*self.par['n_lc']))
					]) 

	return out_vl,out_vu






