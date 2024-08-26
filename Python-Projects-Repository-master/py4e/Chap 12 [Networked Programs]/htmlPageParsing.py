# ############ Chap 12 - Code Examples (urllib1.py, urlwords.py) ############
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
for line in fhand:
    print(line.decode().strip())

# As we have mentioned in the previous script url.py we do not see the headers. 
# The library is handling them for us. 

fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
counts = dict()
for line in fhand:
    print(line.decode().strip())
    words = line.decode().split()
    for word in words:
        counts[word] = counts.get(word, 0) + 1
print(counts)

# ################## Chap 12 - Parsing HTML ##################
# The process of Parsing HTML files is call HTML scraping or HTML spidering.
# When a program or a script pretends to be a browser and retrieves web pages, 
# looks at those web pages, extracts information and then looks at more web pages.
# Search engines scrap web pages - we call this "spidering the web" or "web crawling".

# Not all sites allow web developers to create search engines to crawl their contents.
# You could do string searches the hard way or use the free software library called
# BeautifulSoup from www.crummy.com.

url = input('Enter - ')
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, 'html.parser')

# Retrieve all of the anchor tags.
tags = soup('a')
for tag in tags:
    print(tag.get('href', None))