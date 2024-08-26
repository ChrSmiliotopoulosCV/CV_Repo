from opcode import opname


han = open('mbox-short.txt')

for line in han:
    line = line.rstrip()
    # print('Line:',line)
    # if line == '':
    #     print('Skip Blank')
    #     continue
    wds = line.split()
    # print('Words:', wds)
    # Guardian Pattern
    # if len(wds) < 1:
    #     continue
    # if wds[0] != 'From':
    #     # print('Ignore')
    #     continue
    # print(wds[2])

    # Guardian Pattern in a compound statement.
    if len(wds) < 3 or wds[0] != 'From':
        continue
    print(wds[2])

    # Pay attention the fact that the guardian pattern should be place first and 
    # before the other != statement. Otherwise it will blow.

