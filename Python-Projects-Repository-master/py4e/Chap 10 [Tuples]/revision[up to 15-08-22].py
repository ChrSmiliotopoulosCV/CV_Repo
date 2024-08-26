inputMessage = input("Enter your desired file to open: ")
if len(inputMessage) < 1 : inputMessage = 'clown.txt'
fname = open(inputMessage)
# print(fname)

di = dict()

for lin in fname:
    # print(lin)
    lin = lin.rstrip()
    wds = lin.split()
    print(type(wds))
    for w in wds:
        # idiom: retrieve/create/update counter
        di[w] = di.get(w, 0) + 1

print(di)

tmp = list()

for k, v in di.items():
    print(k, v)
    newtuple = (v, k)
    tmp.append(newtuple)

print(tmp)

rvs = sorted(tmp, reverse=False)
print(rvs)