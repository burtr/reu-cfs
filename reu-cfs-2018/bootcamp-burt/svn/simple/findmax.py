import numpy as np

#
# program to find the maximum integer among a list of integers
#

def findmax(l):
	# l: a list of numbers
	# return the largest number in the list
	# requirement - do this in code.
	
	x = l[0]
	
	#
	# complete
	#
	
	return x


### test program ###

HI_INT = 100

def make_random_list(n):
	return [ np.random.randint(HI_INT) for i in range(n) ]

def test_findmax(t,verbose=False):
	# t the number of trials
	res = []
	for i in range(t):
		a = make_random_list(10)
		x = findmax(a)
		if verbose:
			print("max= {}, list={}".format(x,a))
		res += [x == np.amax(a)]
	return all(res)


if test_findmax(10):
	print ("Success")
else:
	print ("Test failed")
