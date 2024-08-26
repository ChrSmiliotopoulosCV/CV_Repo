# Definite loops use the For keyword. Quite often we have a list of 
# items of the lines in a file - effectively a finite set of things. 
# We can write a loop to run the loop once for each of the items in 
# a set using the Python for construct.

# These loops are called "definite loops" because they execute an 
# exact number of lines.

# We say that "definite loops iterate through the members of a set"

for i in [10,9,8,7,6,5,4,3,2,1,0]:
    print(i)
print("Blastoff!!!")

# Definite loops (for loops) have explicit iteration variables that 
# change each time through a loop. These iteration variables move 
# through the sequence or set.

family = ['Christos', 'Maraki', 'Kleopatra', 'Georgios-Spyridon']

for member in family:
    print("Happy Birthday ", member)
print('Done!!!')