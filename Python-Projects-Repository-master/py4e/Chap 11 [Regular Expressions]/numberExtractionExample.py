import re
hand = open('mbox-short.txt')
numlist = list()
for line in hand:
    line = line.rstrip()
    stuff = re.findall('^X-DSPAM-Confidence: ([0-9.]+)', line)
    if len(stuff) != 1 : continue
    num = float(stuff[0])
    numlist.append(num)
print('Maximum: ', max(numlist))

# ############### Escape Character ###############
# If you want a special regular expression character to just behave 
# normally (most of the time) you prefix it with '\'.

x = 'We just received $10.00 for a loaf of bread.'
y = re.findall('\$[0-9.]+', x)
print(y)

# \$: A real dollar sign
# [0-9.]: A digit or period
# +: At least one or more