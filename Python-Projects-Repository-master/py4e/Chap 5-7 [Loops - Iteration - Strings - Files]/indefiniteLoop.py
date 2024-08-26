# from autopage import line_buffer_from_input


# While loops are called "indefinite loops" because they keep going
# until a logical condition becomes False.
# The loops we have so far are pretty easy to exemine to see if they 
# will terminate or if they will be "indefinite loops".
# Sometimes it is a little harder to be sure if a loop will terminate. 

# while True:
#     line = input("> ")
#     if line == 'done':
#         break
#     print(line)
# print('Done!!!')

while True:
    line = input('> ')
    if line[0] == '#':
        continue
    if line == 'done':
        break
    print(line)
print('Finished. Well Done!!!')