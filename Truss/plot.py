import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcol
import matplotlib.cm as cm

#---------------------------------------------------------------------------------------#
#		Plot of complete structure
#---------------------------------------------------------------------------------------#

def plot_initial(self):

	#figsize
	height = 5*(np.max(self.nodes[:,1])-np.min(self.nodes[:,1])+1)/2
	width  = height*(np.max(self.nodes[:,0])/np.max(self.nodes[:,1]))

	fig,ax = plt.subplots(figsize=(width,height))

	#fixed_nodes
	for num_fixed_node in self.fixed_nodes:
		mount = np.linspace(self.nodes[num_fixed_node,0]-0.1,self.nodes[num_fixed_node,0]+0.1,10)
		ax.fill_between(mount,self.nodes[num_fixed_node,1]-0.1,self.nodes[num_fixed_node,1]+0.1,color='grey')

	#bars
	for bar in self.bars:

		ax.plot([self.nodes[bar[0],0],self.nodes[bar[1],0]],
			[self.nodes[bar[0],1],self.nodes[bar[1],1]],
			color='black')

	#load cases
	for num_node,node in enumerate(self.nodes):
		for num_case in range(self.par['n_lc']):		
			if not np.array_equal(self.load_cases[num_node,:,num_case],np.zeros((self.par['dim']))):

				start = node+0.25*self.load_cases[num_node,:,num_case]
				end   = node

				ax.plot([start[0],end[0]],[start[1],end[1]],color='black')

	#nodes
	ax.plot(self.nodes[:,0],self.nodes[:,1],'o',color='black')

	#axes
	ax.set_xlim(np.min(self.nodes[:,0])-0.2,np.max(self.nodes[:,0])+0.2)
	ax.set_ylim(np.min(self.nodes[:,1])-0.2-0.3,np.max(self.nodes[:,1])+0.2)
	ax.axis('off')

def plot_optimal(self,x):

	#x = [bar_diam,node_disloc]
	bar_diam 	= x[0:self.par['n_b']]
	node_disloc 	= x[-self.par['n_dl']:].reshape(self.par['n_fn'],self.par['dim'],self.par['n_lc'])

	max_line 	= 30
	max_diam 	= 4#np.max(bar_diam)
	threshold_diam 	= 0.01
	disloc_scala 	= 0.025#*np.max(node_disloc)

	#figsize
	height = 5*(np.max(self.nodes[:,1])-np.min(self.nodes[:,1])+1)/2
	width  = height*(np.max(self.nodes[:,0])/np.max(self.nodes[:,1]))

	fig,ax = plt.subplots(figsize=(width,height))

	#fixed_nodes
	for num_fixed_node in self.fixed_nodes:
		mount = np.linspace(self.nodes[num_fixed_node,0]-0.1,self.nodes[num_fixed_node,0]+0.1,10)
		ax.fill_between(mount,self.nodes[num_fixed_node,1]-0.1,self.nodes[num_fixed_node,1]+0.1,color='grey')

	#nodes which are connected
	used_nodes = []

	#bars
	for num_bar,bar in enumerate(self.bars):
		if bar_diam[num_bar] > threshold_diam*max_diam:

			current_width 	= max(threshold_diam,max_line*(bar_diam[num_bar]/max_diam))
			used_nodes.append(bar[0])
			used_nodes.append(bar[1])

			ax.plot([self.nodes[bar[0],0],self.nodes[bar[1],0]],
				[self.nodes[bar[0],1],self.nodes[bar[1],1]],
				color='black',linewidth=current_width,zorder=5)



		elif bar_diam[num_bar] >= 1e20 and bar_diam[num_bar] < threshold_diam*max_diam:

			ax.plot([self.nodes[bar[0],0],self.nodes[bar[1],0]],
				[self.nodes[bar[0],1],self.nodes[bar[1],1]],
				color='black',zorder=5)

		elif bar_diam[num_bar] < 0:

			ax.plot([self.nodes[bar[0],0],self.nodes[bar[1],0]],
				[self.nodes[bar[0],1],self.nodes[bar[1],1]],
				color='red',zorder=5)

	#nodes
	for num_node,node in enumerate(self.nodes):
		if num_node in used_nodes:

			ax.scatter(node[0],node[1],marker='o',s=700,color='black',zorder=10)
			ax.scatter(node[0],node[1],marker='o',s=450,color='white',zorder=10)

		else:
			ax.plot(node[0],node[1],'o',color='black')


	#axes
	ax.set_xlim(np.min(self.nodes[:,0])-0.2,np.max(self.nodes[:,0])+0.2)
	ax.set_ylim(np.min(self.nodes[:,1] + disloc_scala*np.min(node_disloc[:,1]))-0.3,np.max(self.nodes[:,1])+0.2)
	ax.axis('off')

def plot_loaded(self,x,color):

	#x = [bar_diam,node_disloc]
	bar_diam 	= x[0:self.par['n_b']]
	node_disloc 	= np.zeros((self.par['n_n'],self.par['dim'],self.par['n_lc']))

	for num_node,node in enumerate(self.free_nodes):
		node_disloc[node] = x[-self.par['n_dl']:].reshape(self.par['n_fn'],self.par['dim'],self.par['n_lc'])[num_node]

	max_line 	= 30
	max_diam 	= 4#np.max(bar_diam)
	threshold_diam 	= 0.02
	disloc_scala 	= 0.025#*np.max(node_disloc)

	#figsize
	height = 5*(np.max(self.nodes[:,1])-np.min(self.nodes[:,1])+1)/2
	width  = height*(np.max(self.nodes[:,0])/np.max(self.nodes[:,1]))

	if color:
		width += 4

	fig,ax = plt.subplots(figsize=(width,height))

	#fixed_nodes
	for num_fixed_node in self.fixed_nodes:
		mount = np.linspace(self.nodes[num_fixed_node,0]-0.1,self.nodes[num_fixed_node,0]+0.1,10)
		ax.fill_between(mount,self.nodes[num_fixed_node,1]-0.1,self.nodes[num_fixed_node,1]+0.1,color='grey')

	#nodes which are connected
	used_nodes = []

	#stress
	stress = self.stress(x).flatten()

	#user-defined colormap
	cm1 	= mcol.LinearSegmentedColormap.from_list('stress_map',['firebrick','royalblue'])
	#cnorm 	= mcol.Normalize(vmin=min(stress[bar_diam>1e-5]),vmax=max(stress[bar_diam>1e-5]))
	cnorm 	= mcol.Normalize(vmin=-self.par['max_stress'],vmax=self.par['max_stress'])
	cpick 	= cm.ScalarMappable(norm=cnorm,cmap=cm1)
	cpick.set_array([])

	#bars
	for num_bar,bar in enumerate(self.bars):

		#color-coded bars accoriding to stress values
		if color:
			color_bar = cpick.to_rgba(stress[num_bar])

		else:
			color_bar = 'black'

		if bar_diam[num_bar] > threshold_diam*max_diam:

			current_width 	= max(threshold_diam,max_line*(bar_diam[num_bar]/max_diam))
			used_nodes.append(bar[0])
			used_nodes.append(bar[1])

			ax.plot([self.nodes[bar[0],0],self.nodes[bar[1],0]],
				[self.nodes[bar[0],1],self.nodes[bar[1],1]],
				color='grey',linewidth=current_width,zorder=5,alpha=0.5)

			ax.plot([self.nodes[bar[0],0] + disloc_scala*node_disloc[bar[0],0],
				 self.nodes[bar[1],0] + disloc_scala*node_disloc[bar[1],0]],
				[self.nodes[bar[0],1] + disloc_scala*node_disloc[bar[0],1],
				 self.nodes[bar[1],1] + disloc_scala*node_disloc[bar[1],1]],
				color=color_bar,linewidth=current_width,zorder=8)

		elif bar_diam[num_bar] >= 1e20 and bar_diam[num_bar] < threshold_diam*max_diam:

			ax.plot([self.nodes[bar[0],0],self.nodes[bar[1],0]],
				[self.nodes[bar[0],1],self.nodes[bar[1],1]],
				color='grey',zorder=5,alpha=0.5)

			ax.plot([self.nodes[bar[0],0] + disloc_scala*node_disloc[bar[0],0],
				 self.nodes[bar[1],0] + disloc_scala*node_disloc[bar[1],0]],
				[self.nodes[bar[0],1] + disloc_scala*node_disloc[bar[0],1],
				 self.nodes[bar[1],1] + disloc_scala*node_disloc[bar[1],1]],
				color=color_bar,linewidth=current_width,zorder=8)

		elif bar_diam[num_bar] < 0:

			ax.plot([self.nodes[bar[0],0],self.nodes[bar[1],0]],
				[self.nodes[bar[0],1],self.nodes[bar[1],1]],
				color='red',zorder=5)

	#nodes
	for num_node,node in enumerate(self.nodes):
		if num_node in used_nodes:

			ax.scatter(node[0],node[1],marker='o',s=700,color='grey',zorder=7,alpha=0.5)
			ax.scatter(node[0],node[1],marker='o',s=450,color='white',zorder=7)

			ax.scatter(node[0] + disloc_scala*node_disloc[num_node,0],
				   node[1] + disloc_scala*node_disloc[num_node,1],
				   marker='o',s=700,color='black',zorder=10)
			ax.scatter(node[0] + disloc_scala*node_disloc[num_node,0],
				   node[1] + disloc_scala*node_disloc[num_node,1],
				   marker='o',s=450,color='white',zorder=10)

		else:
			ax.plot(node[0],node[1],'o',color='black')

	#colorbar legend
	if color:
		cbar = plt.colorbar(cpick,label='Stress',shrink=0.75)
		cbar.ax.locator_params(nbins=4)
		cbar.ax.tick_params(labelsize=20)
		cbar.ax.set_ylabel('Stress',fontsize=20)

	#axes
	ax.set_xlim(np.min(self.nodes[:,0])-0.2,np.max(self.nodes[:,0])+0.2)
	ax.set_ylim(np.min(self.nodes[:,1] + disloc_scala*np.min(node_disloc[:,1]))-0.3,np.max(self.nodes[:,1])+0.2)
	ax.axis('off')








