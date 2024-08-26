# The while loops are called indefinite loops because they keep 
# going until a logical condition becomes False.
# The loops we used so far are pretty easy to examine if they will 
# terminate or if they will continue to run indefinitely.
# On the other hand, sometimes it is harder to be sure if a loop 
# will terminate.

# n=20
# while n > 0:
#     print(n)
#     n = n - 1
# print("Blastoff")
# print(n)

# while True:
#     line = input("> ")
#     if line == "done":
#         break
#     print(line)
# print("Done!")

# while True:
#     line = input("> ")
#     if line[0] == '#' :
#         continue
#     if line == "done" :
#         break
#     print(line)
# print("Done!")

from audioop import lin2adpcm


while True:
    line = input("> ")
    if line[0] == '#':
        continue
    if line == 'done':
        break
    print(line)
print("Done!!")