# The code in its current form will execute and will through a traceback
# message. Although the code prints the second word of the first line that
# starts with 'From', after that its breaks. This where the developer of the
# code should search to find our what is happening. 

# han  = open('mbox-short.txt')

# for line in han:
#     line = line.rstrip()
#     print('Line: ', line)
#     if line == '':
#         print('Skipped!!!')
#         continue
#     wds = line.split()
#     print('Word: ', wds)
#     # # Guardian sequence
#     # if len(wds) < 1:
#     #     print('Skipped!!!')
#     #     continue
#     if  wds[0] != 'From':
#         print('Ignore')
#         continue
#     print(wds[2])

han  = open('mbox-short.txt')

for line in han:
    line = line.rstrip()
    wds = line.split()
    # # Guardian in a compound statement
    if  len(wds) < 1 or wds[0] != 'From':
        continue
    print(wds[2])