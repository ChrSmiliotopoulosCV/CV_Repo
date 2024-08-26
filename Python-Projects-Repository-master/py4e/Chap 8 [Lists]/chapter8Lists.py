# Algorithms is a set of rules or steps used to solve a problem.
# Data structures is a particular way of organizing data in a computer. You make sure that 
# data what are supposed to do.

# A list is a king of collection which allows us to put many values in a single "variable."

# List constants are surrounded by square brackets and the elements in the list are separated 
# by commas. A list element can be any Python object-even another list. 

# A list can even be empty. 

from turtle import clear


for i in [5,4,3,2,1]:
    print(i)
print('Blastoff!!')

friends = ['Christos', 'Teo', 'Maraki']
print(friends[2])
print(len(friends))
print(range(len(friends)))
print(range(4))

# Pay attention!!! Strings are immutable, whereas lists are mutable objects.

friends = ['Christos', 'Teo', 'Maraki']
for friend in friends:
    print('Happe new year: ', friend)

friends = ['Christos', 'Teo', 'Maraki']
for i in range(len(friends)):
    friend = friends[i]
    print('Happy New Year: ', friend)

# Concatenating lists using +

a = [1,2,3,4,5,6]
b = [7,8,9,10,11,12]

print(a+b)
print(a[3:])

# Remember: Just like in strings, the second number is "up to but not including...."

# There are many more list methods. For example. 

x = list()
print(type(x))
print(dir(x))