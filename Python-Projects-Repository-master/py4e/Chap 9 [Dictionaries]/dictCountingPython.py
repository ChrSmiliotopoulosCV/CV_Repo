# One of the most common usaged for dictionaries is to make 
# histograms. As we have already said one common use of
# dictionaries is counting how ofter we see something. 

# The PROBLEM with the dictionaries is the tracebacks as an error
# to reference a key which is not in the dictionary. We can use the 
# in operator to see if a key is in the dictionary. 

ccc = dict()
print('csev' in ccc)

# In the lines of code that follow we create a histogram code. 
counts = dict()
names = ['csev', 'cwen', 'zqian', 'csev', 'cwen', 'zqian','zqian', 'christos']
for name in names:
    if name not in counts:
        counts[name] = 1
    else:
        counts[name] = counts[name] + 1
print(counts)

# SOS SOS SOS the get method for dictionaries. The pattern of checking to see 
# if a key is already in a dictionary and assuming a default value if the key 
# is not there is so common that there is a method called get() that does this
# for us. It sets a default value if the key does not exist (and no Traceback).

counts = dict()
names = ['csev', 'cwen', 'zqian', 'csev', 'cwen', 'zqian','zqian', 'christos']
for name in names:
    counts[name] = counts.get(name, 0) + 1
print(counts)