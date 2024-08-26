# fruit = 'banana'
# index = 0
# while index < len(fruit):
#     letter = fruit[index]
#     print(index, letter)
#     index = index + 1

# fruit = 'banana'
# index = 0
# for letter in fruit:
#     print(index, letter)
#     index = index + 1
    
fruit = 'banana'
input = input('Enter your letter: ')
if input in fruit:
    print('Found it!!!')
else:
    print('Not found it!!!')