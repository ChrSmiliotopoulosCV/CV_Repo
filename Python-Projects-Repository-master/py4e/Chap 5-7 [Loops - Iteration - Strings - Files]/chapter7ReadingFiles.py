# Before we can read the contents of the file, we must be able to tell Python which file we
# are going to work with and what we will be doing with the file. This is done with the open()
# function. open() returns a "file handle" - a variable use to perform operations on the file. 
# Similar to "File -> Open" in a word file. 

# handle = open(filename, mode)

# mode is optionale and should be 'r' if we are plannint to read the file and 'w' if we are
# planning to write to the file. 

# What is Handle? Handle is a short of a thing to get to the file and not to read the file 
# itself. 

# In python the newline character is introduce via the \n character. 
# It is important to be noted that the \n character is one character and not two. You can check
# it through len(message).
from distutils.fancy_getopt import fancy_getopt
import os


message = 'Hello\nWorld'
print(message)

# File Handle as a sequence.

xfile = open('mbox.txt')
count = 0
for line in xfile:
    # print(line)
    count = count + 1
print('Line Count:', count)

# We can use read() to read the whole file (newlines and all) into a single string. 

# xfile = open('mbox.txt')
# inp = xfile.read()
# print(len(inp))
# print(inp[:50])

# OOOps...everytime the line of a .txt file is printed the print() function 
# adds a new newline in the terminal screen causing some sort of a problem.

# xfile = open('mbox.txt')
# for line in xfile:
#     line = line.rstrip()
#     if line.startswith('From'):
#         print(line)

# or or or or or or we can use the continue statement to search the lines.

# xfile = open('mbox.txt')
# for line in xfile:
#     line = line.rstrip()
#     if not line.startswith('From'):
#         continue
#     print(line)

# There is also the possibility to use in to select lines.

# xfile = open('mbox.txt')
# for line in xfile:
#     line = line.rstrip()
#     if not 'From' in line:
#         continue
#     print(line)

# Pay attention. You can search through various files within
# the same script with just entering the file's name into a 
# variable. Then you execute the script as if any other circumstances.

# In case we want to avoid bad file names as input, we introduce a try and
# except function to handle this issues. 

# fname = input('Please enter file name: ')
# try:
#     xfile = open(fname)
# except:
#     print('Bad file name. The file cannot opened: ', fname)
#     quit()
# for line in xfile:
#     line = line.rstrip()
#     if not 'From' in line:
#         continue
#     print(line)

# or or or or or or 

fname = input('Please enter file name: ')
try:
    xfile = open(fname)
except:
    print('Bad file name. The file cannot opened: ', fname)
    quit()
for line in xfile:
    line = line.rstrip()
    if not line.startswith('From'):
        continue
    print(line)