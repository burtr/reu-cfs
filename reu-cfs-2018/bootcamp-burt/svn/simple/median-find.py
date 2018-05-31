import numpy as np

#
# median-find (k-th largest)
#

def kth_find(loi,k):
	# loi: a list of integers
	# k: a positive integer 1 or greater
	# return the k-th largest element (note: k=1 is smallest number)
	
	assert (len(loi)>0)
	pivot = loi[0]
	lower, upper = [], []

	# separate loi into lower and upper according to comparison to pivot

	# write code

	assert(len(lower)+len(upper)+1 == len(loi))

	# either pivot is the k-th largest or recurse on lower or upper elements,
	# adjusting the value k if needed

	# write code

	return 0 

def median_find(loi):
	return kth_find(loi,len(loi)//2)


### test program ###

HI_INT = 100

def make_random_list(n):
	return [ np.random.randint(HI_INT) for i in range(n) ]

def test_kth_find(t,verbose=False):
	# test the kth_find def
	a = [ i for i in range(30) ]
	
	# test for the case of ascending numbers in the list
	if verbose:
		print(a)
	for i in range(30):
		if verbose:
			print("the {}-th element is {}".format(i+1,kth_find(a,i+1)))
		else:
			assert(kth_find(a,i+1)==i)

	# test for the case of descending numbers in the list
	a = [ 29-i for i in range(30) ]
	if verbose:
		print(a)
	for i in range(30):
		if verbose:
			print("the {}-th element is {}".format(i+1,kth_find(a,i+1)))
		else:
			assert( i == kth_find(a,i+1))

	# test for min and max in a randomly generated list
	list_len = 22
	for i in range(t):
		a = make_random_list(list_len)
		min_is = kth_find(a,1)
		if min_is != np.amin(a):
			return False
		max_is = kth_find(a,list_len)
		if max_is != np.amax(a):
			return False
		if verbose:
			print("min={}, \tmax={},\t list={}".format(min_is,max_is,a))
	return True


if test_kth_find(10,verbose=True):
	print ("Success")
else:
	print ("Test failed")


