
import numpy as np 

# 
# sample program about slices, masking, copys and views
#
# author: burt
# created: 23 may 2018
# last update:


print("masking makes a copy")
a = np.array([ 3, 4, 6, 1, 9, 8, 2 ])
abig = a>5
print (a, abig, a[abig]) 
a[abig][1] = -1
print (a, abig, a[abig]) 

print("numpy slices are views (by reference)")
a = np.array([ 3, 4, 6, 1, 9, 8, 2 ])
print(a, a[::-1])
a[::-1][1] = -1
print(a, a[::-1])

print("numpy slices are views (by reference)")
a = np.array([ 3, 4, 6, 1, 9, 8, 2 ])
print(a, a[::2])
a[::2][2] = -1
print(a, a[::2])

print("python lists are slices (by copy)")
a = [ 3, 4, 6, 1, 9, 8, 2 ]
print(a, a[::2])
a[::2][2] = -1
print(a, a[::2])

print("python lists are slices (but copy is shallow)")
a = [ 3, 4, [6], 1, 9, 8, 2 ]
print(a, a[::-1])
a[::-1][4][0] = -1
print(a, a[::-1])


print("a view of every other element is made by manipulating the stride, not by copy")
a = np.array([ 3, 4, 6, 1, 9, 8, 2 ])
b = a[::2].view()
print(a,b)
print(a.strides,b.strides)
a[0] = -1
print(a,b)

 

