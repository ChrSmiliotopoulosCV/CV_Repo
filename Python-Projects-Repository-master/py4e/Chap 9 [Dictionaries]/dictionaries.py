# What is a Collection?
# A collection is nice because we can put more than one value in it 
# and carry them all around in one conventional package.
# We have a bunch of values in a single "variable". We do this by 
# having more than one place "in" the variable. We have ways of finding 
# the different places in the variable. 

# Variables are not collections since they take only one value and every new 
# value is overwritten above the old one. 

# A list is a linear organised form of a collection, whereas a dictionary is 
# its messy equivalen where all data are flagged with a value. 

# Lists index their entries based on the position in the list. Dictionaries are 
# like bags - no order at all. So we index the thing we put in the dictionary 
# with a 'lookup tag'. 

purse = dict()
purse['money'] = 12
purse['candy'] = 3
purse['tissues'] = 75
purse['money'] = purse['money'] + 3
print(purse)
purse['money'] = 20
print(purse)

age = dict()
age = {'Christos' : 36, 'Teo' : 33, 'Ilias' : 30, 'Maria' : 34, 'Nick' : 30}
print(age)

lst = list()
lst.append(22)
lst.append(55)
lst.append(39)
print(lst)
lst[0] = 24
print(lst)
print(lst[0])