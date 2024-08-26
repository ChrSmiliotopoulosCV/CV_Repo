# ################### Regular Expessions ###################
# In computer a regular expression, also referred to as "regex"
# or "regexp", provides a concise and flexible means of matching
# strings of text, such as particulaf characters, words or patterns
# of characters. A regular expression is written in a formal 
# laguage that can be interpreted by a regular expression processor.

# It is a much more characterized and really clever "wild card" expressions
# language for matching and parsing strings. 

# They are mostrly used for smart "Find" or "Search".

# ################### Understanding Regular Expessions ###################
# Very powerfull and quite cryptic.
# Fun once you uderstand them.
# Regular expressions are a language unto themselves.
# A language of "marker characters" - programming with characters.
# It is a kind of "old school" language - compact.

# Summary of the most commonly use "Regular Expressions."

# While this only scratched the surface of regular expressions, we have learned 
# a bit about the language of regular expressions. They are search strings with 
# special characters in them that communicate your wishes to the regular expression 
# system as to what defines “matching” and what is extracted from the matched strings. 
# Here are some of those special characters and character sequences:

# ^ Matches the beginning of the line.

# $ Matches the end of the line.

# . Matches any character (a wildcard).

# \s Matches a whitespace character.

# \S Matches a non-whitespace character (opposite of \s).

# * Applies to the immediately preceding character(s) and indicates to match zero or more times.

# *? Applies to the immediately preceding character(s) and indicates to match zero or more times 
# in “non-greedy mode”.

# + Applies to the immediately preceding character(s) and indicates to match one or more times.

# +? Applies to the immediately preceding character(s) and indicates to match one or more times 
# in “non-greedy mode”.

# ? Applies to the immediately preceding character(s) and indicates to match zero or one time.

# ?? Applies to the immediately preceding character(s) and indicates to match zero or one time 
# in “non-greedy mode”.

# [aeiou] Matches a single character as long as that character is in the specified set. In this 
# example, it would match “a”, “e”, “i”, “o”, or “u”, but no other characters.

# [a-z0-9] You can specify ranges of characters using the minus sign. This example is a single 
# character that must be a lowercase letter or a digit.

# [^A-Za-z] When the first character in the set notation is a caret, it inverts the logic. This 
# example matches a single character that is anything other than an uppercase or lowercase letter.

# ( ) When parentheses are added to a regular expression, they are ignored for the purpose of 
# matching, but allow you to extract a particular subset of the matched string rather than the 
# whole string when using findall().

# \b Matches the empty string, but only at the start or end of a word.

# \B Matches the empty string, but not at the start or end of a word.

# \d Matches any decimal digit; equivalent to the set [0-9].

# \D Matches any non-digit character; equivalent to the set [^0-9].

# You can also use re.search() to see if a string matches a regular expression, similar to using the 
# find() method for strings. There is also the re.findall() regular expression to extract portions of 
# a string that match you regular expression, similar to a combination of find() and slicing: var[5:10].

import re

# Our initial chunk of code without any regular expression.
hand = open('mbox-short.txt')
for line in hand:
    line = line.rstrip()
    # if line.find('From:') >= 0:
    #     print(line)
    if re.search('From:', line):
        print(line)
    if line.startswith('From:'):
        print(line)
    if re.search('^From:', line):
        print(line)

# The difference in the two approaches is that in the first we are looking for a method and the other
# we are programming with regular expressions. 

# ################### Wild-Card Expessions ###################
# The dot character matches any character.
# If you add the asterisk character, the character is "any number of times".
# e.g. "^X.*"
# e.g. "^X-\S+"

# ################### Matching and Extracting with Regular Expressions ###################
# As we have already mentioned re.search() returns a True/False depending on whether the string matches
# the regular expression. 

# If we actually want the matching strings to be extracted, we use re.findall().

x = 'My 2 favorite numbers are 19 and 42'
# The + in the brackets means one or more digits.
y = re.findall('[0-9]+', x)
print(y)

# ################### Warning: Greedy Matching ###################
# The repeat characters (* and +) push outward in both directions (greedy) to match the largest
# possible string.

x = 'From: Using the : character'
# The + in the brackets means one or more digits.
y = re.findall('^F.+:', x)
print(y)
# When it has a choice it chooses the largest one. It gets "Greedy"

# We can transform the code to be not greedy. 
x = 'From: Using the : character'
# The + in the brackets means one or more digits.
y = re.findall('^F.+?:', x)
print(y)

# Fine-Tuning String Extraction
x = 'From chr.smiliotopoulos@gmail.com Tue Aug 16 11:29:10 2022'
# The + in the brackets means one or more digits.
y = re.findall('\S+@\S+', x)
print(y)
# Parentheses are not part of the match - but they tell where to start and stop 
# what string to extract.
y = re.findall('^From (\S+@\S+)', x)
print(y)

