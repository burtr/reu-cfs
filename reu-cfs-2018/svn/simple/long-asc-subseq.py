import numpy as np

#
# program to find the longest (strictly) ascending subsequence
#


# https://people.cs.clemson.edu/~bcdean/dp_practice/dp_3.swf


def findmax_indx(l):
	# l: a list of numbers
	# return the index of largest number in the list
	# requirement - do this in code.
	
	return 0


def longest_ascending_subsequence(a):
	# a: a list of integers
	
	l = [0] * len(a)
	p = [0]	* len(a)
	
	# L.I. for i<j, l[i] is the length of the long asc s/seq ending in a[j]
	# and p[i] is the predecessor of a[i] in that s/seq, if p[i]==i end of subseq

	l[0] = 1
	p[0] = 0
	
	# complete
	
	return []


### test program ###

HI_INT = 100

def make_random_list(n):
	return [ np.random.randint(HI_INT) for i in range(n) ]

def check_ascend(l):
	res = [ l[i]<l[i+1] for i in range(len(l)-1)]
	return all(res)



def test_longest_ascending_subsequence(t):

	seq_3 = ([(3*i)%10 for i in range(10)],4)
	seq_2 = ([(2*i)%10 for i in range(10)],5)
	van_der_corput = ([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15],6)

	lass = longest_ascending_subsequence(seq_3[0])
	if len(lass)!=seq_3[1]:
		return False
	lass = longest_ascending_subsequence(seq_2[0])
	if len(lass)!=seq_2[1]:
		return False
	lass = longest_ascending_subsequence(van_der_corput[0])
	if len(lass)!=van_der_corput[1]:
		return False
	for i in range(t):
		a = make_random_list(20)
		lass = longest_ascending_subsequence(a)
		if not check_ascend(lass):
			return False
	return True


if test_longest_ascending_subsequence(10):
	print ("Success")
else:
	print ("Test failed")
