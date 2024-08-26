# Best friends: Strings and Lists
# split() breaks a sting into parts and produces a list of strings. We think of these
# as words. We can access a particular word or loop through all the words.

import this
from turtle import st

abc = 'Hello world, I would like to introduce myself. My name is Christos and I am working and active Officer in Hellenic Army, and more precisely as a Sperwer Mission Planner for the UAV batallion of Greek land forces.'
stuff = abc.split()
print(stuff)
print(len(stuff))
print(stuff[0])

# You can always loop within a list just to print each word included.

for w in stuff:
    print(w)

# SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS SOS
# This is very important for the Machine learning algorithm scripts. How set delimiters to 
# eliminate unwanted characters from a .csv file. 

# When you do not specify a delimiter, multiple spaces are treated like one delimiter.
# However, you can specify what delimiter character to use in the splitting. 

line = 'christos;maraki;kleopatra;georgios-spyridon;margarita;fondas-nickos'
thing = line.split()
print(len(thing))
thing1 = line.split(';')
print(thing1)
print(len(thing1))


