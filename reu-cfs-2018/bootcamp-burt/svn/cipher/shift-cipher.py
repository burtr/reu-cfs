import string
import sys
import os
import argparse

#
# shift-cipher.py
#
# author:bjr
# date: may 2018
# last update:
#

args_g = 0  # args are global

def shift_encipher(p,k):
	"""
	input: 
		p: string of lowercase letters, the plain text
		k: the key, a letter
	return:
		ciphertext, a string of uppercase letters
	"""
	c = ""
	kord = ord(k)-ord('a') 
	
	# code

	return c ;

def shift_decipher(c,k):
	"""
	input: 
		c: string of uppercase letters, the cipher text
		k: the key, a letter
	return:
		plaintext, a string of lowercase letters
	"""
	p = ""
	kord = ord(k)-ord('a') 

	# code

	return p ;


def parse_args():
	parser = argparse.ArgumentParser(description="Encrypt/decrypt stdin by a shift cipher. Ignores any character other than alphabetic.")
	parser.add_argument("key", help="encipherment key")
	parser.add_argument("-d", "--decrypt", action="store_true", help="decrypt, instead of encrypting")
	parser.add_argument("-g", "--word_group", type=int, default=5, help="characters per word group")
	parser.add_argument("-G", "--line_group", type=int, default=5, help="word groups per line")
	parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
	return parser.parse_args()

def main(argv):

	global args_g
	args_g = parse_args()
	assert (len(args_g.key)==1)

	## gather plain text and format
	pt = ""
	for line in sys.stdin:
		for c in line:
			if c.isalpha():
				if not args_g.decrypt:
					c = c.lower()
				else:
					c = c.upper()
				pt += c
 
	## encrypt/decrypt
	if args_g.decrypt:
		ct = shift_decipher(pt,args_g.key)
	else:
		ct = shift_encipher(pt,args_g.key)

	## pretty print ct
	i = 0
	s = ""
	r = args_g.word_group * args_g.line_group

	for c in ct:
		s += c
		i += 1
		if i%r==0:
			s += '\n'
		else:
			if i%args_g.word_group==0:
				s += ' '

	print (s)
	

main(sys.argv)
