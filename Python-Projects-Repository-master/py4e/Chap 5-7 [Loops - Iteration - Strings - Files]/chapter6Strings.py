# String data type. A string litaral uses quotes 'Hello'
# or "Hello". For strings, + means "concatenate". When a 
# string contains a number, it is still a string. We can 
# convert number in a string into a number using int(). 

from dataclasses import dataclass
from itertools import count


str1 = 'Hello'
str2 = "there"
bob = str1 + str2
print(bob)

apple = input('Enter your number: ')
x = int(apple) - 10
print(x)

# We can get at any single character in a string using an index
# specified in square brackets. The index value must be an integer
# and starts at zero. The index can be an expression that is computed.

# banana
# 012345

fruit = 'banana'
letter = fruit[2]
print(letter)

# String have length too...!!!

fruit = 'mango'
print(len(fruit))

# ######################### Looping Through Strings #########################
# Using a while statement and an iteration variable and the len() function, we
# can contruct a loop to look at each of the letters in a string individually.

# ######################### Indefinite While Loop #########################

fruit = 'banana'
index = 0
while index < len(fruit):
    letter = fruit[index]
    print(index, letter)
    index = index + 1

# ######################### Definite For Loop #########################

fruit = 'banana'
for letter in fruit:
    print(letter)

# ######################### Looping and Counting #########################

# The loop that follows is a simple loop that loops through each letter in a 
# string and counts the number of times the loop encounters the 'a' character.

# As we can see in the example that follows. The iteration variable "iterates" 
# through the string and the block (body) of code is executed once for each 
# value in the sequence. 

word = 'banana'
count = 0
for letter in word:
    if letter == 'a':
        count = count + 1
print(count)

# More string operators - Slicing Strings.
# We can also look at any continuous sections of a string using a colon operator.
# The second number is one beyond the end of the slice - "up to but not including".
# If the second number is beyond the end of the string it stopss at the end.

# #### SOS SOS SOS To this point we should remember..."up to but not included" ####

word = "Monty Python"
index = 0
for letter in word:
    if index < len(word):
        print(index, letter)
        index = index + 1
print(word[6:20])
print(word[6:7])
print(word[0:4])

# Slicing Strings - If we leave off the first number or the last number of the slice, it
# is assumed to be the beginning or end of the string respectively.

print(word[3:7])

# String Concatenation
a = 'Hello'
# b = a + ' World'
b = a + ' ' + 'World'
print(b)

# Using in as a logical Operator. The in keyword can also be used to check to see if one 
# string is "in" another string. The in expression is a logical expression that returns True
# or False and can be used in an if statement. 

fruit = 'banana'
input = input('Enter your letter: ')
if input in fruit:
    print('Found it!!!')
else:
    print('Not found it!!!')

# Python has a number of string functions which are in the string library. These functions
# are already built into every string - we invoke them by appending the function to the string
# variable. These functions do not modify the original string, instead they return a new string
# that has been altered. 

greet = 'Hello Hob'
zap = greet.lower()
print(zap)
print(type(zap))
print(dir(zap))

# https://docs.python.org/3/library/stdtypes.html string-methods

greet = "           Hello Bob      "
print(greet.strip())

# Parsing and Extracting

data = 'From chr.smiliotopoulos@gmail.com Sat 30 Jul 22 07:04:22'
index = 0
for letter in data:
    print(index, letter)
    index = index + 1
atpos = data.find('@')
print(atpos)

sppos = data.find(' ', atpos)
print(sppos)

host = data[atpos+1 : sppos]
print(host)

# There are two kinds of Strings from Python 2 to Python 3. One of the big 
# thing in Python 3 is that all the internal characters are Unicode.



