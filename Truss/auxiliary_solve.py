import numpy as np

#---------------------------------------------------------------------------------------#
#		Options ipopt
#---------------------------------------------------------------------------------------#

def add_option_ipopt(self,problem):

	if len(self.options_ipopt) > 0:

		for option in self.options_ipopt:
			if len(option) == 1:
				problem.addOption(option)
			elif len(option) == 2:
				problem.addOption(option[0],option[1])

#---------------------------------------------------------------------------------------#
#		Iteration update ALM
#---------------------------------------------------------------------------------------#

def print_stat_alm(self):

	print()
	print('----------------------------------')
	print('Iter\t',self.par_ALM['iter']+1)
	print('alpha\t',self.par_ALM['alpha'])
	print('eta\t',self.par_ALM['eta'])
	print('----------------------------------')
	print()