# ################ Sorting a Dictionary using Tuples ################
from opcode import opname


inputMessage = input('Enter file: ')
if len(inputMessage) < 1 : inputMessage = 'clown.txt'
fname = open(inputMessage)

di = dict()

for lin in fname:
    lin = lin.rstrip()
    wds = lin.split()
    for w in wds:
        # idiom: retrieve/create/update counter
        di[w] = di.get(w, 0) + 1

# print(di)

# for k, v in di.items():
    # print(k, v)

print(di.items())

# x = sorted(di.items())
# print(x[:5])

# The previous two lines of code sort the dictionary entries based on the 
# key and not based on the value itself. In the lines of code that follow
# we well manually construct the list. 

tmp = list()
for k, v in di.items():
    print(k, v)
    newtuple = (v, k)
    tmp.append(newtuple)

print('Flipped: ', tmp)

tmp = sorted(tmp)
print("Sorted: ", tmp)

tmp = sorted(tmp, reverse=True)
print("Sorted-Reversed: ", tmp[:5])

# Now if we want to print the first five elements in a smother way, we make a loop.
for v,k in tmp[:5]:
    print(k,v)


