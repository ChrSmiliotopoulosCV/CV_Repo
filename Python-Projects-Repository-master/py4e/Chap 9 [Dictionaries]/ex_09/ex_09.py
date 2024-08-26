fname = input('Enter File: ')
if len(fname) < 1 : fname = 'clown.txt'
hand = open(fname)

di = dict()
for line in hand: 
    lin = line.strip()
    # print(lin)
    wds = line.split()
    # print(wds)
    for w in wds:
        # if the key is not there there the count is zero
        # print(w)
        # if w in di:
        #     di[w] = di[w] + 1
        #     print('**Existing**')
        # else:
        #     di[w] = 1
        #     print('**NEW**')
        # print(di[w])

        # All the previous lines are replaced with the one following
        # statement.

        # idiom: retrieve/create/update counter
        di[w] = di.get(w,0) + 1

# print(di)

# now we want to find the most common word
largest = -1
theword = None
for k,v in di.items():
    # print(k,v)
    if v > largest:
        largest = v
        theword = k # capture/remember the key that was largest

print('Done', theword, largest)