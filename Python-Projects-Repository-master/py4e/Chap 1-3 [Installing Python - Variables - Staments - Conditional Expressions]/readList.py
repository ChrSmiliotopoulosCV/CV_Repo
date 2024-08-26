a_file = open("filters.txt", "r")

lines = a_file.read()
list_of_lists = lines.splitlines()

a_file.close()

print(list_of_lists)

x = 'Get-PassHashes'
counter = 0

print(lines)

for i in list_of_lists:
    if x == i:
        print("Evreka!!!")
        counter += 1 
        print(counter)
        countvar = counter
print(countvar, " Lateral Movement events have been found in total!!!")