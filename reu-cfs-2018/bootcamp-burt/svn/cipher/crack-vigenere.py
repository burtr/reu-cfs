import string
import sys
import os
import argparse
import numpy as np

#
# crack-vigenere.py
#
# author:
# date: 
# last update:
#

args_g = 0  # args are global

def frequencycount(s):
	"""
	input:
		s: string of lowercase letters
	return:
		an integer array of length 26, where index i has the count of occurances of the i-th letter
		(where a is the 0th letter and z is the 25th letter)
	"""
	count = np.zeros(26, dtype=int)

	# code

	return count


def get_statistics(filename):
	f = open(filename,"r")
	p = "" ;
	for line in f:
		for c in line :
			if c.isalpha() :
				p = p + c.lower() ;
	f.close() ;
	return frequencycount(p) ;


	#
	# code
	#


def parse_args():
	parser = argparse.ArgumentParser(description="Cracks a vigenere cipher by freqency analysis, given the key length.")
	parser.add_argument("key_length", type=int, help="the presumed length of the encipherment key")
	parser.add_argument("reference_text", help="a text file sampling the letter frequence statistics")
	parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
	return parser.parse_args()

def main(argv):

	global args_g
	args_g = parse_args()

	fc = get_statistics(args_g.reference_text)
	if args_g.verbose:
		print fc

	## gather plain text and format
	t_in = ""
	for line in sys.stdin:
		for c in line:
			if c.isalpha():
				t_in += c

	if args_g.verbose:
		print t_in

	#
	# code
	#

	print "password"


main(sys.argv)
