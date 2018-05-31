import string
import sys
import os
import argparse
import numpy as np
import math
import fractions as frac

#
# ioc.py
#
# author:
# date: 
# last update:
#


args_g = 0  # args are global

def parse_args():
	parser = argparse.ArgumentParser(description="Calculate index of coincidence over stdin, writing result to stdout.")
	parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
	return parser.parse_args()

def main(argv):

	global args_g
	args_g = parse_args()

	## gather input
	t = ""
	for line in sys.stdin:
		for c in line:
			if c.isalpha():
				t += c

	#
	# code
	# 
	
	key_length_guess = 0
	print key_length_guess
	

main(sys.argv)