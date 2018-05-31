import numpy as np

#
# program to selection sort integers
#

def findmin_indx(l):
	# l: a list of numbers
	# return the index of largest number in the list
	# requirement - do this in code.

	
	x = l[0]
	idx = 0
	
	# complete
	
	return idx

def selection_sort(l):
	# l: a list of integers
	# return the list l sorted from smallest to largest integer
	
	s = []
	while len(l)>0:
	
	# complete
		del l[0] # this is get the template to compile and halt
		
	return s



### test program ###

HI_INT = 100

def make_random_list(n):
	return [ np.random.randint(HI_INT) for i in range(n) ]

def test_selection_sort(t,verbose=False):
	# t the number of trials
	res = []
	for i in range(t):
		a = make_random_list(10)
		a_s = sorted(a)
		b = selection_sort(a)
		if verbose:
			print ("list: {},\t sorted:{}".format(a_s,b))
		res += [a_s==b]
	return all(res)


if test_selection_sort(10):
	print ("Success")
else:
	print ("Test failed")
