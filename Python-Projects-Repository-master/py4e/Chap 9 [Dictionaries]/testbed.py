# christDict = dict()
# print('ccc' in christDict)

# counts = dict()
# names = ['csev', 'cwen', 'zqian', 'csev', 'cwen', 'zqian','zqian', 'christos', 'maraki']

# for name in names:
#     if name not in counts:
#         counts[name] = 1
#     else:
#         counts[name] = counts[name] + 1
# print(counts)

# for name in names:
#     counts[name] = counts.get(name, 0) + 1
# print(counts)

from tkinter.messagebox import NO

# name = input('Enter filename: ')
# handle = open(name)

# counts = dict()
# for line in handle:
#     words = line.split()
#     for word in words:
#         counts[word] = counts.get(word, 0) + 1

# # print(counts.items())

# bigcount = None
# bigword = None

# for word, count in counts.items():
#     if bigcount is None or count > bigcount:
#         bigword = word
#         bigcount = count

# print(bigword, bigcount)

name = input('Enter your filename: ')
handle = open(name)

counts = dict()
for line in handle:
    words = line.split()
    for word in words:
        counts[word] = counts.get(word, 0) + 1
print(counts.items())

bigcount = None
bigword = None

for word, count in counts.items():
    if bigcount is None or count > bigcount:
        bigword = word
        bigcount = count

print(bigword, bigcount)