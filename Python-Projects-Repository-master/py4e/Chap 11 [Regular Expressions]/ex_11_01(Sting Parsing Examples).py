# ############## Regular Expressions - String Parsing Examples ##############
from cgi import print_exception
from cmath import pi
import email
import re

data = 'From chr.smiliotopoulos@ou.ac.uk Tue Aug 16 11:29:10 2022'

# The example that follows is a typical example of extracting a host name - using 
# find and sting slicing.
atpos = data.find('@')
print(atpos)

sppos = data.find(' ', atpos)
print(sppos)

hostname = data[atpos+1 : sppos]
print(hostname)

# The double split pattern. This kind of pattern is much cleaner than the previously 
# presented typical example of sting slicing. 
words = data.split()
print(words)
email = words[1]
print(email)
pieces = email.split('@')
print(pieces)
print(pieces[1])

# The Regex Version of the double split pattern.
y = re.findall('@([^ ]*)', data)
print(y)
# [^ ]: Match non-blank character.
# *: Match many of them.
# ([^ ]*): All the statement is used to Extract the non-blank characters.

# Even Cooler Regex Version
y = re.findall('^From .*@([^ ]*)', data)
print(y)

