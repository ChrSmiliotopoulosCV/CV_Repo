# The so called Definite Loops are called as such because they execute an
# exact - definite number of times. We say that "definite loops iterate through
# the members of a set".

# "i" is called iteration variable, which iterates through the sequence. The block of 
# the code is executed once for each value in the sequence. The iteration variable
# moves through all of the values in the sequence. 

from cgitb import small
import this
from tkinter.messagebox import NO


for i in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,19,20]:
    print(i)
print('Blastoff!!!')

# ################### Making 'smart' loops ###################
# ################ What is the largest number? ###############

largest_so_far = -1
print('Before', largest_so_far)
for the_num in [2 ,3, 5, 6, 7, 22, 45, 57, 77, 26]:
    if the_num > largest_so_far:
        largest_so_far = the_num
    print(largest_so_far, the_num)
print('After', largest_so_far)

# ################### Chapter 5: More Loop Patterns ###################
# ######################### Counting in a Loop ########################

# An average just combines the counting and sum patterns and divides when 
# the loop is done.

count = 0
sum = 0
print('Before', count, sum)
for value in (9, 41, 55, 12, 3, 46, 77, 21, 269, 44, 335, 31):
    count = count + 1
    sum = sum + value
    print(count, sum, value)
print('After', count, sum, sum / count)

# We use an if statement in the loop to catch / filter the values we are
# looking for. 

print('Before')
for value in (9, 41, 55, 12, 3, 46, 77, 21, 269, 44, 335, 31):
    if value > 50:
        print('Large Number', value)
print('After')

# If we just want to search and know

found = False
print('Before', found)
for value in [9, 41, 55, 12, 3, 46, 77, 21, 269, 44, 335, 31]:
    if value == 77:
        found = True
    print(found, value)
print('After', found)

# How to find the smallest value
# ################### Making 'smart' loops ###################
# ################ What is the smallest number? ###############
# How would we change this to make it find the smallest value 
# in the list? We could change "largest_so_far" to "smallest_so_far" 
# and > to <........Is it going to fix it? Check it on cmd....
# ################### Wrong!!!! ###################

smallest_so_far = -1
print('Before', smallest_so_far)
for the_num in [2 ,3, 5, 6, 7, 22, 45, 57, 77, 26]:
    if the_num < smallest_so_far:
        smallest_so_far = the_num
    print(smallest_so_far, the_num)
print('After', smallest_so_far)

# The solution is hidden in creating a loop that captures the first 
# value, assume to be the smallest_so_far and continue after that.
# ################### SOS SOS SOS SOS SOS SOS SOS!!!! ###################
# The == operator compares the value or equality of two objects, whereas 
# the Python is operator checks whether two variables point to the same 
# object in memory. In the vast majority of cases, this means you should 
# use the equality operators == and !=, except when you’re comparing to None.

# https://www.geeksforgeeks.org/difference-between-and-is-operator-in-python/

# Python has an is operator that can be used in logical expressions. Implies "
# "is the same as". Similar to, but stronger than ==. is not also is a logical 
# operator. 

smallest = None
print('Before')
for value in [2 ,3, 5, 6, 7, 22, 45, 57, 77, 26]:
    if smallest is None:
        smallest = value
    elif value < smallest:
        smallest = value
    print(smallest, value)
print('After', smallest)
