import string
import sys
import os
import argparse
import numpy as np

#
# crack-shift.py
#
# author:
# date: 
# last update:
#

args_g = 0  # args are global

def frequencycount(s):
	"""
	input:
		string of lowercase letters
	return:
		an integer array of length 26, where index i has the count of occurances of the i-th letter
		(where a is the 0th letter and z is the 25th letter)
	"""
	count = np.zeros(26, dtype=int)
	
	# code

	return count

def dotproduct_twist(s,t,twist):
	"""
	input:
		s, t array of integer
		twist, and integer
	return:
		the dot product between s and t when t is "rolled forward" by the amount twist
	"""

	# code
	
	return  0

def crack_shift(ct,rt):
	"""
	input:
		ct: the cipher text, lower case letters
		rt: the reference test, lower case letters (for frequency count)
	return:
		the shift key that was used to encode the cipher text	
	"""

	# code

	return "a"

def get_statistics(filename):
	f = open(filename,"r")
	p = "" ;
	for line in f:
		for c in line :
			if c.isalpha() :
				p = p + c.lower() ;
	f.close() ;
	return frequencycount(p) ;

def parse_args():
	parser = argparse.ArgumentParser(description="Cracks a shift cipher by freqency analysis.")
	parser.add_argument("reference_text", help="a text file sampling the letter frequence statistics")
	parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
	return parser.parse_args()

def main(argv):

	global args_g
	args_g = parse_args()

	fc = get_statistics(args_g.reference_text)
	if args_g.verbose:
		print (fc)

	## gather plain text and format
	t_in = ""
	for line in sys.stdin:
		for c in line:
			if c.isalpha():
				t_in += c

	x = crack_shift(fc,frequencycount(t_in))
	print (x)

main(sys.argv)

