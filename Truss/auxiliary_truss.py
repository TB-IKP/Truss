from itertools import combinations
from scipy.spatial.distance import euclidean

import numpy as np
import numpy.ma as ma

#---------------------------------------------------------------------------------------#
#		Construction of all possible bars for given nodes
#---------------------------------------------------------------------------------------#

def potential_bars(self):
	'''Determination of all potential for the given nodes.'''

	out_bars = []

	#get all 2-tuples of nodes
	tuples_indices = list(combinations(range(len(self.nodes)),2))

	for bar in tuples_indices:

		if bar[0] in self.fixed_nodes and bar[1] in self.fixed_nodes:
			#no bar between two fixed nodes
			continue

		elif euclidean(self.nodes[bar[0]],self.nodes[bar[1]]) > self.par['max_length']:
			#no bar if length exceeds max value
			continue

		else:
			#check if bar overlaps with other bar
			#i.e. if another node is located on the bar

			#minimum and maximum coordinates of nodes of current bar
			min_bound = np.min(np.vstack((self.nodes[bar[0]],self.nodes[bar[1]])),axis=0)
			max_bound = np.max(np.vstack((self.nodes[bar[0]],self.nodes[bar[1]])),axis=0)

			#find coordinates which do not change
			nonzero_dim = np.nonzero(self.nodes[bar[0]]-self.nodes[bar[1]])[0]
			mask = [True]*len(self.nodes)

			for dim in range(self.par['dim']):

				if dim in nonzero_dim:
					mask = mask & \
						ma.masked_greater(self.nodes[:,dim],min_bound[dim]).mask& \
						ma.masked_less(self.nodes[:,dim],max_bound[dim]).mask
				else:
					mask = mask & \
						ma.masked_equal(self.nodes[:,dim],self.nodes[bar[0],dim]).mask

			#nodes which are located "in between" end nodes of current bar
			crit_nodes = self.nodes[mask]

			if len(crit_nodes) > 0: 
			
				#test whether node between end nodes is indeed on the bar

				for node in crit_nodes:

					#calculate dist(node_1,node_2)-dist(node_1,crit_node)-dist(node_2,crit_node)
					diff_dist = euclidean(self.nodes[bar[1]],self.nodes[bar[0]]) \
						- euclidean(self.nodes[bar[0]],node) \
						- euclidean(self.nodes[bar[1]],node)

					if diff_dist < 1e-9:
						continue
					else:
						out_bars.append(bar)

			else:
				out_bars.append(bar)

	return np.array(out_bars)

#---------------------------------------------------------------------------------------#
#		Bar lengths
#---------------------------------------------------------------------------------------#

def bar_lengths(self):
	'''Determination of the bar lengths for the given ground structure.'''

	out_bar_lengths = np.array([euclidean(self.nodes[bar[1]],self.nodes[bar[0]]) for bar in self.bars])

	return out_bar_lengths

#---------------------------------------------------------------------------------------#
#		Bar angles
#---------------------------------------------------------------------------------------#

def bar_angles(self):
	'''Determination of the bar angles for the given ground structure
	with repect to the displacement coordinate system.'''

	out_bar_angles = np.zeros((len(self.nodes),self.par['dim'],len(self.bars)))

	for num_bar,bar in enumerate(self.bars):

		if bar[0] not in self.fixed_nodes:
			out_bar_angles[bar[0],:,num_bar] = -(self.nodes[bar[1]]-self.nodes[bar[0]])/self.bar_lengths[num_bar]

		if bar[1] not in self.fixed_nodes:
			out_bar_angles[bar[1],:,num_bar] = -(self.nodes[bar[0]]-self.nodes[bar[1]])/self.bar_lengths[num_bar]

	return out_bar_angles

#---------------------------------------------------------------------------------------#
#		Stiffness matrix
#---------------------------------------------------------------------------------------#

def stiffness_matrix(self,x):
	'''Determination of the stiffness matrix K(x).

	Parameters:
	-----------
	x: array
		Bar diameters and nodal displacements as obtained from solve() method.
	'''

	#x = [bar_diam,node_disloc]
	bar_diam 	= x[0:self.par['n_b']]

	out_stiff_mat 	= np.zeros((self.par['n_fn']*self.par['dim'],
				self.par['n_fn']*self.par['dim']))

	for num_bar in range(self.par['n_b']):
		out_stiff_mat += bar_diam[num_bar]*self.par['E']/self.bar_lengths[num_bar] * \
			self.bar_angles[self.free_nodes,:,num_bar].reshape(self.par['n_fn']*self.par['dim'],1) * \
			self.bar_angles[self.free_nodes,:,num_bar].reshape(1,self.par['n_fn']*self.par['dim'])

	return out_stiff_mat

#---------------------------------------------------------------------------------------#
#		Stress sigma
#---------------------------------------------------------------------------------------#

def stress(self,x):
	'''Determine stress on the individual bars.

	Parameters:
	-----------
	x: array
		Bar diameters and nodal displacements as obtained from solve() method.
	'''

	#x = [bar_diam,node_disloc]
	bar_diam 	= x[0:self.par['n_b']]
	node_disloc 	= x[-self.par['n_dl']:].reshape(self.par['n_fn'],self.par['dim'],self.par['n_lc'])

	out_stress = np.zeros((self.par['n_lc'],self.par['n_b']))

	for num_case in range(self.par['n_lc']):
		for num_bar in range(self.par['n_b']):

			out_stress[num_case,num_bar] = self.par['E']/self.bar_lengths[num_bar] * \
							np.dot(self.bar_angles[self.free_nodes,:,num_bar].reshape(self.par['n_fn']*self.par['dim']),
							node_disloc[:,:,num_case].reshape(self.par['n_fn']*self.par['dim']))

	return out_stress










