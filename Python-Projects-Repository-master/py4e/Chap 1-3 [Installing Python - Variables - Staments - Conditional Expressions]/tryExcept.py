from email.errors import InvalidBase64CharactersDefect


astr = 'Hello Bob'
try:
    astr = int(astr)
except:
    istr = -1

print('First', istr)

astrs = '1234'
try:
    astr = int(astrs)
except:
    istrs = -1

print('Second', astrs)

rawstr = input('Enter a number: ')
try:
    ival = int(rawstr)
except:
    ival = -1

if ival > 0:
    print('Nice Work')
else:
    print('Not a number')