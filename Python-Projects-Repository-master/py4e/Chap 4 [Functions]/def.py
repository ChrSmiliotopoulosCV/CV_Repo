def thing():
    print('Hello')
    print('Fun')


thing()
print('.zip')
thing()

print(int('1234') + 1)

def greet(lang):
    if lang == 'es':
        # print('Hola')
        return 'Hola'
    elif lang == 'fr':
        # print('Bonjour')
        return 'Bonjour'
    else:
        # print('Hello')
        return 'Hello'

# greet('es')
# greet('fr')
# greet('en')

print(greet('es'), 'Christos')
print(greet('fr'), 'Christos')
print(greet('en'), 'Christos')

def greeting():
    return 'Hello'

print(greeting(), 'Chris')

def multtwo(a, b):
    multiply = a * b
    return multiply
 
print(multtwo(5, 6))

