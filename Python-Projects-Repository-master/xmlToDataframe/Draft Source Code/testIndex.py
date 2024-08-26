# my_dict = {'Australia' : 200, 'Germany' : 100, 'Europe' : 300, 'Switzerland' : 400}
  
# find_key = 'Germany'
# print("dictionary is : " + str(my_dict))
# res = list(my_dict.keys()).index(find_key)
# print("Index of find key is : " + str(res))

from itertools import count
from pyexpat.errors import XML_ERROR_DUPLICATE_ATTRIBUTE
from tkinter.messagebox import NO
import numpy as np
import pandas as pd

def add_value(dict_obj, key, value = None):
    ''' Adds a key-value pair to the dictionary.
        If the key already exists in the dictionary, 
        it will associate multiple values with that 
        key instead of overwritting its value'''
    if key not in dict_obj:
        if value is not None:
            # dict_obj[key] = value
            # This is a draft attempt to initialize a new key with the zero value and append eventually the imported value.
            # dict_obj[key] = '0'
            # dict_obj[key].append(value)
            # SOS SOS SOS SOS SOS SOS SOS the problem with pandas is when there is only one value on the key-value set it is not
            # recognised as a dictionary list. Here we correct this bug with this little chunk of code and finally we get the result we 
            # want even for new column names.
            dict_obj[key] = ['0']
            dict_obj[key] = [value]
        else:
            dict_obj[key] = ['0', 'NaN']
    elif isinstance(dict_obj[key], list):
        if value is not None:
            dict_obj[key].append(value)
        else:
            dict_obj[key].append('NaN')
    else:
            if value is not None:
                dict_obj[key] = [dict_obj[key], value]
            else:
                dict_obj[key] = value

# List of keys
keyList = ['chris', 'Maraki', 'George', 'Liakos', 'Jonas', 'Vagos', 'Juarez']
my_dict = dict()
keyList1 = ['Name', 'Guid', 'EventID', 'Version', 'Level', 'Task', 'Opcode', 'Keywords', 'SystemTime', 'EventRecordID', 'Correlation', 'ProcessID', 'ThreadID', 'Channel', 'Computer', 'UserID', 'RuleName', 'UtcTime', 'ProcessGuid', 'ProcessId', 'Image', 'FileVersion', 'Description', 'Product', 'Company', 'OriginalFileName', 'CommandLine', 'CurrentDirectory', 'User', 'LogonGuid', 'LogonId', 'TerminalSessionId', 'IntegrityLevel', 'Hashes', 'ParentProcessGuid', 'ParentProcessId', 'ParentImage', 'ParentCommandLine', 'ChristosssCommandLine']
# for i in keyList:
#     my_dict[i] = ['0', 'NaNNaN']
      
print(my_dict)

for v in my_dict.values():
    print(len(v))

add_value(my_dict, 'chris')
add_value(my_dict, 'Maraki')
add_value(my_dict, 'George')
# add_value(my_dict, 'chris', 'newline')
# add_value(my_dict, 'chris', 33)
# add_value(my_dict, 'Maraki', 34)
# add_value(my_dict, 'George')
# add_value(my_dict, 'chris', 37)
# add_value(my_dict, 'Maraki', 29)
# # add_value(my_dict, 'Maraki')
# # add_value(my_dict, 'Maraki')
# add_value(my_dict, 'George', 22)
# # add_value(my_dict, 'chris', 25)
# add_value(my_dict, 'Liakos')
add_value(my_dict, 'Jonas', 12)
add_value(my_dict, 'Liakos', 21)
add_value(my_dict, 'Juarez')

# print(my_dict)
# keys_list = list(my_dict)
# key = keys_list[0]
# print(key)

# max_len = max(len(l) if isinstance(l, list) else 1 for l in my_dict.values())
# print(max_len)

# for key in my_dict.keys():
#     if len(my_dict[key]) != max_len:
#       christos = my_dict[key]
#       my_dict[key] = np.repeat('NaN', max_len-1).tolist()
#       my_dict[key].append(christos)

print(my_dict)

for v in my_dict.values():
    print(len(v))

# print(my_dict)

# add_value(my_dict, 'chris' , 29)
# add_value(my_dict, 'Maraki', 69)
# add_value(my_dict, 'chris' , 29)
# add_value(my_dict, 'Maraki', 69)

# print(my_dict)

max_len = max(len(l) if isinstance(l, list) else 1 for l in my_dict.values())
print('The Max length is: ', max_len)
for key in my_dict.keys():
    while len(my_dict[key]) < max_len:
    #   christos = my_dict[key]
    #   my_dict[key] = np.repeat('NaN', max_len-1).tolist()
      my_dict[key].append('OMG')

print('The new advanced dictionary is: ', my_dict)

# for v in my_dict.values():
#     print(len(v))

# # add_value(my_dict, 'Maraki', 19)
# # add_value(my_dict, 'Liakos', 37)
# add_value(my_dict, 'Jonas', 22)
# add_value(my_dict, 'Liakos', 21)
# # add_value(my_dict, 'Jonas', 43)
# add_value(my_dict, 'Vagos', 49)
# # add_value(my_dict, 'Spyros', 57)
# # add_value(my_dict, 'Giannis', 63)

# max_len = max(len(l) if isinstance(l, list) else 1 for l in my_dict.values())
# print('The Max length is: ', max_len)
# for key in my_dict.keys():
#     while len(my_dict[key]) < max_len:
#     #   christos = my_dict[key]
#     #   my_dict[key] = np.repeat('NaN', max_len-1).tolist()
#       my_dict[key].append('NaN')

# print(my_dict)

# for v in my_dict.values():
#     print(len(v))

add_value(my_dict, 'chris', 55)
add_value(my_dict, 'chris', 56)
add_value(my_dict, 'Maraki', 54)
add_value(my_dict, 'Maraki', 55)
add_value(my_dict, 'George', 22)
add_value(my_dict, 'George', 23)
add_value(my_dict, 'Jonas')
add_value(my_dict, 'Jonas', 11)
add_value(my_dict, 'Liakos', 54)
add_value(my_dict, 'Liakos', 58)
add_value(my_dict, 'Juarez')
add_value(my_dict, 'CCosta', 333)

# print(my_dict)

# for v in my_dict.values():
#     print(len(v))

max_len = max(len(l) if isinstance(l, list) else 1 for l in my_dict.values())
print('The Max length is: ', max_len)
for key in my_dict.keys():
    while len(my_dict[key]) < max_len:
    #   christos = my_dict[key]
    #   my_dict[key] = np.repeat('NaN', max_len-1).tolist()
      my_dict[key].append('OMG')

print(my_dict)

for v in my_dict.values():
    print(len(v))

add_value(my_dict, 'CCosta', 567)

max_len = max(len(l) if isinstance(l, list) else 1 for l in my_dict.values())
print('The Max length is: ', max_len)
for key in my_dict.keys():
    while len(my_dict[key]) < max_len:
    #   christos = my_dict[key]
    #   my_dict[key] = np.repeat('NaN', max_len-1).tolist()
      my_dict[key].append('OMG')

print(my_dict)

for v in my_dict.values():
    print(len(v))

add_value(my_dict, 'chris', 55)
add_value(my_dict, 'Maraki', 54)
add_value(my_dict, 'George', 22)
add_value(my_dict, 'Jonas')
add_value(my_dict, 'Liakos', 58)
add_value(my_dict, 'Juarez')
add_value(my_dict, 'CCosta', 333)

max_len = max(len(l) if isinstance(l, list) else 1 for l in my_dict.values())
print('The Max length is: ', max_len)
for key in my_dict.keys():
    while len(my_dict[key]) < max_len:
    #   christos = my_dict[key]
    #   my_dict[key] = np.repeat('NaN', max_len-1).tolist()
      my_dict[key].append('OMG')

print(my_dict)

for v in my_dict.values():
    print(len(v))

add_value(my_dict, 'CCosta', 982)

max_len = max(len(l) if isinstance(l, list) else 1 for l in my_dict.values())
print('The Max length is: ', max_len)
for key in my_dict.keys():
    while len(my_dict[key]) < max_len:
    #   christos = my_dict[key]
    #   my_dict[key] = np.repeat('NaN', max_len-1).tolist()
      my_dict[key].append('OMG')

print(my_dict)

for v in my_dict.values():
    print(len(v))

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in my_dict.items()]))

# This peace of code will be use during the last step of the execution of this script
# and it is dedicated in the extraction (df.drop(index=0)) of the dataframe's row with 
# index=0. This row is used only during the initialiation phase of the converter and is
# removed towards the final extractin of the .csv file. 

# df = df.drop(index=0)

# df = pd.DataFrame.from_dict(my_dict, orient='index')

print(df)

# computing number of rows
rows = len(df.axes[0])
 
# computing number of columns
cols = len(df.axes[1])
 
print("Number of Rows: ", rows)
print("Number of Columns: ", cols)

# count = { k: len(v) for k, v in my_dict.items() }

# for key in my_dict.keys():
#     count = len(my_dict[key])
#     print(count)
#     christos = my_dict[key]
#     my_dict[key] = np.repeat('NaN', count-1).tolist()
#     my_dict[key].append(christos)

# print(my_dict)

# for v in my_dict.values():
#     list1 = []
#     list1.append(v)
#     print(type(list1))
#     print(list1)

# d = { "a":[1,2,3], "b":[1,2,3,4,5], "c":[1,2], "d":[1,2,3,4,5,6,7] }  

# count = { k: len(v) for k, v in d.items() }

# print(count)

# # List of keys
# keyList = ["Chris", "Maraki"]
  
# # initialize dictionary
# d = {}
  
# # iterating through the elements of list
# for i in keyList:
#     d[i] = 'NaN'

# print(d)

