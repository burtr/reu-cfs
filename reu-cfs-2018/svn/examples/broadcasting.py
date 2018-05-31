import numpy as np

# 
# sample program about numpy broadcasting
#
# author: burt
# created: 23 may 2018
# last update:


print ("")
print ("")
a = np.array([ 3, 4, 6, 1, 9, 8, 2 ])
b = np.array([5])
print ("a.shape= {}\na= {}\nb.shape= {}\nb= {}".format(a.shape,a,b.shape,b))
print("a+b=")
print (a+b)

print ("")
print ("")
a = np.array([[1,2,3],[4,5,6]])
print ("a.shape= {}\na= {}\nb.shape= {}\nb= {}".format(a.shape,a,b.shape,b))
print("a+b=")
print (a+b)

print ("")
print ("")
a = np.array([[1,2,3],[4,5,6]])
b = np.array([10,20,30]) 
print ("a.shape= {}\na= {}\nb.shape= {}\nb= {}".format(a.shape,a,b.shape,b))
print("a+b=")
print (a+b)



print("")
print("")
a = np.array([1,0,1])
b = np.array([[1],[0],[1]])
print("a.shape= {}\na={}".format(a.shape,a))
print("b.shape= {}\nb={}".format(b.shape,b))
print("a+b=")
print(a+b)
print("b+a=")
print(b+a)

print("\n\n")
a = np.arange(27).reshape((3,3,3))
b = np.arange(9).reshape((3,3))
c = a + b
print(c)
print(c.reshape((27)))


print("\n\n")
a = np.array([1,0,1]) 
b = a.reshape((3,1))
c = a.reshape((3,1,1))
print("a.shape= {}\na= {}\nb.shape= {}\nb= {}\nc.shape= {}\nc= {}\n".format(a.shape,a,b.shape,b,c.shape,c))
print("a+b+c=")
print(a+b+c)
