# ############# Using urllib in Python #############
# Since HTTP is so common, we have a library that does all the socket work 
# for us and makes web pages look like a file. 

import imp
from itertools import count
import urllib.request, urllib.parse, urllib.error

fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
for line in fhand:
    print(line.decode().strip())

# When the lines of code are executeed and the romeo.txt is appeared in the terminal's
# screen you will notice that the headers that were previously appeared with sockets are 
# disappeared. urlopen() is responsible for that and although there is a way to make the 
# headers visible, right now we will stick to this form of execution. 

# ############# Treat the url with urllib as a file... #############
import urllib.request, urllib.parse, urllib.error

fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
counts = dict()
for line in fhand:
    print(line.decode().strip())
    words = line.decode().split()
    for word in words:
        counts[word] = counts.get(word, 0) + 1
print(counts)

# ################# Reading Web Pages #################
import urllib.request, urllib.parse, urllib.error
fhand = urllib.request.urlopen('http://www.dr-chuck.com/page1.htm')
for line in fhand:
    print(line.decode().strip())
