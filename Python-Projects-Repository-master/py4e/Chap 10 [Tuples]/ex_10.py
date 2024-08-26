# In Chapter 10 we are going to talk for our third king of collection,
# which is Tuples. Tuples are like lists. They are another kind of sequence
# that functions much like a list - they have elements which are indexed at 0.

# This is a list.
from re import X
from tkinter import Y

x = []
print(type(x))
x.append('Christos')
x.append('Maraki')
x.append('Kleopatra')
x.append('Georgios-Spyridon')
print(x[2])
x[2] = 'Kleopatraki'
print(x)

# Now we will create a Tuple. 
# SOS SOS Tuples are "immutable". Unlike the lists, once you create 
# a tuple, you cannot alter its contents - similar to a string. 
# For example:

y = ('Christos', 'Maraki', 'Kleopatra', 'Georgios-Spyridon')
print(type(y))
print(y)

# The reason why tuples are "immutable" is efficiency. Anything that
# a developer can do to modify a list is not allowed for tuples. Just 
# compare the functions include in list() and the relevants for tuple().

# l = list()
# print('List functions: ', dir(l))

# t = tuple()
# print('Tuple functions: ', dir(t))

# It is obvious that tuples are limited lists. So why do we like them?
# The only reason is efficiency when we need to create a collection that 
# Python wont never let us modify. They are prefered when temporary variables
# are prefered to be used. 

# ################### Tuples and Assignements. ###################

# We can also put a tuple on the left-hand side of an assignement statement. 
# We can even omit the parenthesis.

(x,y) = (4, 'Christos')
print(y)

(a,b) = (99, 98)
print(a)

# or we can also omit the parenthesis...
c,d = 97,96
print(c)

# ################### Tuples and Dictionaries. ###################

# Tuples are connected to dictionaries. The items() method in dict()
# returns a list of (key, value) tuples. As in the example that follows:

d = dict()
d['Christos'] = 36
d['Maraki'] = 33
d['Kleopatra'] = 9
d['Georgios-Spyridon'] = 7

for k,v in d.items():
    print(k,v)

tups = d.items()
print(type(tups))
print(tups)

# ################### Tuples are Comparable. ###################

# The comparison operators work with tuples and other sequences. If the first
# item is equal, Python goes on to the next element, and so on, until it finds 
# elements that differ.

print((0,1,2) < (5,1,2))

print((0,1,2000000) < (0,1,5))

# ######################### Tuples Sorting #########################
# ################### Sorting Lists of Tuples... ###################

# We can take advantage of the ability to sort a list of tuples to get a sorted
# version of a dictionary.

# First we sort the dictionary by the key using the item() method and sorted() 
# function.

d = dict()
d['a'] = 10
d['c'] = 22
d['b'] = 1

# D.items() -> a set-like object providing a view on D's items
print(d.items())

# Return a new list containing all items from the iterable in ascending order.
print(sorted(d.items()))

# We can do this even more directly using the built-in function sorted() that 
# takes a sequence as a parameter and returns a sorted sequence.

d = {'b':1, 'd':44, 'a':10, 'c':22, 'e':29}
print(d)
for k, v in sorted(d.items()):
    print(k, v)

# ################### Sort by values instead of key. ################### 
# If we could construct a list of tuples of the form (value, key) we could
# sort by value.
# We could do this with a for loop that creates a list of tuples.

c = {'b':1, 'd':44, 'a':10, 'c':22, 'e':29}
tmp = list()

for k, v in c.items():
    tmp.append((v, k))
print(tmp)

tmp = sorted(tmp, reverse=True)
print(tmp)





