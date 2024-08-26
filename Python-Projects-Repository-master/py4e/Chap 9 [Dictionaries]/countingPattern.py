from itertools import count


counts = dict()
line = input('Enter a line of text: ')
words = line.split()

print('Words: ', words)
print('Counting....')

for word in words:
    counts[word] = counts.get(word, 0) + 1

print('Counts', counts)

# We can create a definite for loop to make the algorithm iterate over
# the keys and their values and print the results. 

for key in counts:
    print(key, counts[key])

# Retrieving lists of keys and values with pre-configures methods.

print(list(counts))
print(counts.keys())
print(counts.values())
print(counts.items())

for aaa,bbb in counts.items():
    print(aaa, bbb)