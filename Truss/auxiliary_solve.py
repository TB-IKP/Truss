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