
inputMessage = input('Enter your file to open: ')
fhand = open(inputMessage)
counts = dict()

for line in fhand:
    line = line.rstrip()
    words = line.split()
    for word in words:
        counts[word] = counts.get(word, 0) + 1

lst = list()
for key, val in counts.items():
    newtup = (val, key)
    lst.append(newtup)

lst = sorted(lst, reverse=True)

for val, key in lst[:10]:
    print(key, val)

# What can be said about the previous example is that it is very procedural 
# and algorithms and data-structures like to solve the desired problem.

# ################### This is an Even Shorter Version. ################### 
c = {'b':1, 'd':44, 'a':10, 'c':22, 'e':29}

print(sorted([(v, k) for k, v in c.items()]))

# List comprehension creates a dynamic list. In this case, we make a list of 
# reserved tuples and then sort it.


