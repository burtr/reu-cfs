import numpy as np

# a program to find two integers with a given difference, given
# a set of integers

# hint: first sort the set of integers. then maintain two indices, i <= j,
# as j moves forward the gap between l[i] and l[j] increases, and as
# i moves forward the gap between l[i] and l[j] decreases.
# now search by moving either i or j forward according to whether the
# gap is more or less than the desired difference


def separated_int(l,dist):
	# given a list of integers l, and a distance, dist
	# find two numbers l[i], l[j] with l[i]-l[j]==d, 
	# or return None
	
	l_s = sorted(l)
	i, j = 0 , 0
	while j<len(l):

		# complete
		return (0,0) # this is an example, and to get it to compile

	return None


### test program ###

HI_INT = 100

def make_random_list(n):
	return [ np.random.randint(HI_INT) for i in range(n) ]

def test_separated_int(t):
	tl = [2,6,9,5,11]
	for i in [1,2,3]:
		int_pair = separated_int(tl,i)
		if (int_pair[1]-int_pair[0]) != i:
			return False
	for i in range(t):
		tl = make_random_list(20)
		int_pair = separated_int(tl,4)
		if int_pair:
			if (int_pair[1]-int_pair[0]) != 4:
				return False
	return True




if test_separated_int(10):
	print ("Success")
else:
	print ("Test failed")
	
	