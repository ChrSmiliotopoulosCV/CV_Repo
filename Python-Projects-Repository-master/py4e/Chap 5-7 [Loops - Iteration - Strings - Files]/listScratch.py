# We can create an empty list and then add elements using the append method.
# The list stays in order and new elements are added at the end of the list. 

from matplotlib.style import available


stuff = list()
stuff.append('book')
stuff.append(99)
stuff.append('cookie')
print(stuff)

# Is something in a list operator...

print(99 in stuff)
print(20 in stuff)
print(30 not in stuff)

# Lists could be put in order....
friends = list()
friends.append('Christos')
friends.append('Josephine')
friends.append('Vincent')
friends.append('Hulu')
print(friends)
friends.sort()
print(friends)
print(friends[2])

nums = [2,5,6,7,8,22,3,54,68,38,81,99]
print(max(nums))
print(min(nums))
print(len(nums))
print(sum(nums))
print(sum(nums) / len(nums))


# How about creating a loop to make our sum and average calculations.....
# While loop...
# total = 0
# count = 0
# while True:
#     imp = input('Enter a number: ')
#     if imp == 'done' : break
#     value = float(imp)
#     total = total + value
#     count = count + 1
# average = total / count
# print('Average: ', average)

# Now we will do the same but with the list() function
numlist = list()
while True:
    imp = input('Enter a number: ')
    if imp == 'done' : break
    value = float(imp)
    numlist.append(value)
print(numlist)
average = sum(numlist) / len(numlist)
print('Average: ', average)