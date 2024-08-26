inp = input("Europe Floor?")

# SOS always input() gives back a number as a String and not as an Integer.
# You have to convert the input() number to Int() in order to be used.

usf = int(inp) + 1

print("US Floor is ", usf)
