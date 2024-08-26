# About Characters and Strings...
# American Standard Code for Information Interchange (ASCII)
# Representing Simple Strings with Python....
# Each character is represented by a number between 0 and 256 stored in
# 8 bits (1 byte) of memory. In python we can find the representation of 
# each character with the ord() function which tells us the numeric value 
# of a simple ASCII character.

print(ord('H'))
print(ord('e'))
print(ord('\n')) # Newline character is the number 10

# Nowdays ASCII was replaced with Unicode 9.0 with so much space to represent 
# any character someone might wants.
# UTF-8 (1-4 bytes) is recommended practice for encoding data to be exchanged between 
# computing systems.

# In Python 3 all strings are Unicode.

# ################# Python Strings to Bytes #################
# When we talk to an external resource like a network socket we send bytes, so we need 
# to encode Python 3 strings into a given character encoding. 
# When we read data from an external resource, we must decode it based on the character 
# set so it is properly represented in Python3 as a string. 

