# HOW MAKE "SMART" LOOPS.
# The trick is "knowing" something about the whole loop when you 
# are struck writing code that only sees one entry at a time. 
# Set some variables to initial values.
# for thing in data:
# Look for something or do something to each entry separetely, 
# updating a variable. Then, look at the variables.

print("Before")
for thing in [9, 41, 12, 3, 5, 17]:
    print(thing)
print("After")